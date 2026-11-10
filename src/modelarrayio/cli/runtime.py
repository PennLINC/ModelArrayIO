"""Runtime policy, transactional publication, and cheap conversion metrics."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import tiledb

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable only on Windows
    resource = None

from modelarrayio.cli import utils as cli_utils
from modelarrayio.utils.loading import (
    LazyScalarMatrices,
    resolve_loader_workers,
    validate_workers,
)

logger = logging.getLogger(__name__)

__all__ = [
    'TimingRecorder',
    'atomic_output',
    'load_timing_metrics',
    'path_stats',
    'peak_rss_bytes',
    'publish_scalar_outputs',
    'resolve_loader_workers',
    'scalar_output_stats',
    'validate_hdf5_output',
    'validate_tiledb_output',
    'validate_timing_jsonl_path',
    'validate_workers',
]


def load_timing_metrics(load_stats) -> dict[str, float | int]:
    """Translate staging counters into consistently named runtime metrics."""
    values = load_stats.to_dict() if load_stats is not None else {}
    source_wall = float(values.get('source_wall_seconds', 0.0))
    staging_wall = float(values.get('wall_seconds', 0.0))
    staging_cpu = float(values.get('cpu_seconds', 0.0))
    return {
        'source_load_seconds': source_wall,
        'source_worker_seconds': float(values.get('source_seconds', 0.0)),
        'validation_seconds': float(values.get('validation_seconds', 0.0)),
        'staging_seconds': staging_wall,
        'staging_cpu_seconds': staging_cpu,
        'staging_cpu_utilization_percent': (
            100.0 * staging_cpu / staging_wall if staging_wall else 0.0
        ),
        'staging_allocation_seconds': float(values.get('allocation_seconds', 0.0)),
        'scheduler_wait_seconds': float(values.get('scheduler_wait_seconds', 0.0)),
        'in_flight_sources_mean': float(values.get('in_flight_sources_mean', 0.0)),
        'in_flight_sources_max': int(values.get('in_flight_sources_max', 0)),
        # Rows served by a reader fast path (CIFTI raw data-block read); a
        # value below source_count means some rows silently fell back.
        'fast_path_rows': int(values.get('fast_path_rows', 0)),
    }


def _tiledb_runtime_settings() -> dict[str, int | str]:
    """Return the TileDB thread settings without modifying its global pools."""
    config = tiledb.Config()
    settings: dict[str, int | str] = {}
    for metric_name, config_name in (
        ('tiledb_compute_concurrency_level', 'sm.compute_concurrency_level'),
        ('tiledb_io_concurrency_level', 'sm.io_concurrency_level'),
    ):
        value = config[config_name]
        try:
            settings[metric_name] = int(value)
        except (TypeError, ValueError):
            settings[metric_name] = str(value)
    return settings


def _open_file_descriptor_count() -> int | None:
    """Return the current descriptor count on Linux/macOS when available."""
    for descriptor_dir in (Path('/proc/self/fd'), Path('/dev/fd')):
        try:
            return sum(1 for _ in descriptor_dir.iterdir())
        except OSError:
            continue
    return None


def _temporary_sibling(output: Path) -> Path:
    token = uuid.uuid4().hex
    return output.with_name(f'.{output.name}.modelarrayio-tmp-{token}')


def validate_timing_jsonl_path(
    jsonl_path: str | Path | None,
    output_paths: Sequence[str | Path],
    *,
    directory_outputs: bool,
) -> None:
    """Reject telemetry paths that would be replaced by or corrupt an output."""
    if jsonl_path is None:
        return
    timing_path = Path(jsonl_path).resolve(strict=False)
    for output_path in output_paths:
        resolved_output = Path(output_path).resolve(strict=False)
        conflicts = timing_path == resolved_output
        if directory_outputs:
            conflicts = conflicts or resolved_output in timing_path.parents
        if conflicts:
            raise ValueError(
                f'Timing JSONL path {str(jsonl_path)!r} conflicts with conversion output '
                f'{str(output_path)!r}; choose a path outside the output.'
            )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _is_tiledb_internal(path: Path) -> bool:
    """Return True for TileDB's own bookkeeping entries (``__group``, ``__meta``, ...)."""
    return path.name.startswith('__')


def _is_tiledb_array(path: Path) -> bool:
    return tiledb.object_type(str(path)) == 'array'


def _directory_merge_entries(temporary: Path, output: Path) -> list[str]:
    """Return the relative paths of *temporary* to move into the existing *output* root.

    A top-level entry that *output* lacks moves as a whole. A top-level group
    that both roots have (normally ``scalars`` and ``parcels``) is merged one
    level down, so each scalar group or parcel array replaces only its
    namesake and every other scalar already in *output* is kept. TileDB
    bookkeeping entries are moved only where *output* has none of its own.
    """
    entries: list[str] = []
    for child in sorted(temporary.iterdir()):
        target = output / child.name
        if _is_tiledb_internal(child):
            if not (target.exists() or target.is_symlink()):
                entries.append(child.name)
            continue
        if (
            child.is_dir()
            and target.is_dir()
            and not _is_tiledb_array(child)
            and not _is_tiledb_array(target)
        ):
            for grandchild in sorted(child.iterdir()):
                existing = target / grandchild.name
                if _is_tiledb_internal(grandchild) and (
                    existing.exists() or existing.is_symlink()
                ):
                    continue
                entries.append(f'{child.name}/{grandchild.name}')
        else:
            entries.append(child.name)
    return entries


def _merge_directory(temporary: Path, output: Path) -> None:
    """Move each entry of *temporary* into *output*, rolling every move back on failure.

    Replaced entries are parked in a same-filesystem sibling backup directory
    until all moves succeed, so an interrupted publication leaves *output*
    exactly as it was.
    """
    entries = _directory_merge_entries(temporary, output)
    backup_root = output.with_name(f'.{output.name}.modelarrayio-backup-{uuid.uuid4().hex}')
    moved: list[tuple[Path, Path, Path | None]] = []
    try:
        for entry in entries:
            source = temporary / entry
            target = output / entry
            backup = None
            if target.exists() or target.is_symlink():
                backup = backup_root / entry
                backup.parent.mkdir(parents=True, exist_ok=True)
                logger.warning('Replacing existing %s in TileDB root %s', entry, output)
                target.replace(backup)
            moved.append((source, target, backup))
            source.replace(target)
    except BaseException:
        restored = True
        for source, target, backup in reversed(moved):
            try:
                if not (source.exists() or source.is_symlink()) and (
                    target.exists() or target.is_symlink()
                ):
                    target.replace(source)
                if backup is not None and (backup.exists() or backup.is_symlink()):
                    backup.replace(target)
            except OSError:
                restored = False
                logger.exception(
                    'Could not restore %s during rollback; the previous content is kept in %s.',
                    target,
                    backup_root,
                )
        if restored:
            _remove_path(backup_root)
        raise
    _remove_path(backup_root)


def _publish_directory(temporary: Path, output: Path) -> None:
    """Publish a TileDB root: rename it into place, or merge into an existing root.

    Merging matches the long-standing behaviour of converting one scalar at a
    time into the same ``.tdb`` root: arrays written by this run replace their
    namesakes, and every other scalar already in the root is kept.
    """
    if not output.exists():
        try:
            temporary.replace(output)
            return
        except OSError:
            # Another process published the root first; merge into it instead.
            if not output.is_dir():
                raise
    _merge_directory(temporary, output)
    _remove_path(temporary)


@contextmanager
def atomic_output(output: str | Path, *, directory: bool):
    """Yield a same-parent temporary path and publish it only on success.

    Files replace any existing output. Directories (TileDB roots) are merged
    into an existing root by :func:`_publish_directory`.
    """
    final_path = Path(output)
    if directory and final_path.exists() and not final_path.is_dir():
        raise NotADirectoryError(
            f'TileDB output {str(final_path)!r} exists and is not a directory.'
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(final_path)
    _remove_path(temporary)
    try:
        yield temporary
        if not temporary.exists():
            raise RuntimeError(f'Conversion did not create temporary output {temporary}.')
        if directory:
            _publish_directory(temporary, final_path)
        else:
            os.replace(temporary, final_path)
    except BaseException:
        _remove_path(temporary)
        raise


def _normalized_names(values) -> list[str]:
    """Return storage strings as ordinary Python text."""
    array = np.asarray(values).reshape(-1)
    return [
        value.decode('utf-8') if isinstance(value, (bytes, bytearray, np.bytes_)) else str(value)
        for value in array
    ]


def _expected_element_count(expected_elements, scalar_name: str) -> int | None:
    if expected_elements is None:
        return None
    if isinstance(expected_elements, Mapping):
        value = expected_elements.get(scalar_name)
        return int(value) if value is not None else None
    return int(expected_elements)


def validate_hdf5_output(
    path: str | Path,
    scalar_names: Sequence[str],
    *,
    expected_sources: Mapping[str, Sequence[str]] | None = None,
    expected_dtype: str | np.dtype | None = None,
    expected_elements: int | Mapping[str, int] | None = None,
    required_paths: Sequence[str] = (),
) -> None:
    """Validate scalar shape, dtype, source order, and required metadata."""
    dtype = np.dtype(expected_dtype) if expected_dtype is not None else None
    with h5py.File(path, 'r') as h5_file:
        for required_path in required_paths:
            if required_path not in h5_file:
                raise RuntimeError(
                    f'Output validation failed: missing required HDF5 path {required_path!r}.'
                )
        for scalar_name in scalar_names:
            values_path = f'scalars/{scalar_name}/values'
            columns_path = f'scalars/{scalar_name}/column_names'
            if values_path not in h5_file or columns_path not in h5_file:
                raise RuntimeError(
                    f'Output validation failed: missing {values_path!r} or {columns_path!r}.'
                )
            values = h5_file[values_path]
            columns = h5_file[columns_path]
            if not isinstance(values, h5py.Dataset) or values.ndim != 2:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'{values_path!r} must be a two-dimensional dataset.'
                )
            if not isinstance(columns, h5py.Dataset) or columns.ndim != 1:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'{columns_path!r} must be a one-dimensional dataset.'
                )
            if values.shape[0] != columns.shape[0]:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'{values.shape[0]} rows but {columns.shape[0]} source names.'
                )
            if dtype is not None and values.dtype != dtype:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'dtype {values.dtype} != expected {dtype}.'
                )
            element_count = _expected_element_count(expected_elements, scalar_name)
            if element_count is not None and values.shape[1] != element_count:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'{values.shape[1]} elements != expected {element_count}.'
                )
            stored_sources = _normalized_names(columns[...])
            if expected_sources is not None:
                expected = [str(value) for value in expected_sources[scalar_name]]
                if stored_sources != expected:
                    raise RuntimeError(
                        f'Output validation failed for scalar {scalar_name!r}: '
                        'source names or order do not match the cohort.'
                    )
            if values.shape[0] and values.shape[1]:
                # Force representative first/last chunks through the normal
                # decoder before publication without reading the whole matrix.
                values[0, 0]
                values[-1, -1]


def validate_tiledb_output(
    path: str | Path,
    scalar_names: Sequence[str],
    *,
    require_column_name_arrays: bool = False,
    expected_sources: Mapping[str, Sequence[str]] | None = None,
    expected_dtype: str | np.dtype | None = None,
    expected_elements: int | Mapping[str, int] | None = None,
    required_paths: Sequence[str] = (),
) -> None:
    """Validate TileDB scalar shape, dtype, source order, and metadata."""
    root = Path(path)
    dtype = np.dtype(expected_dtype) if expected_dtype is not None else None
    for required_path in required_paths:
        required_uri = str(root / required_path)
        if tiledb.object_type(required_uri) is None:
            raise RuntimeError(
                f'Output validation failed: missing required TileDB path {required_uri!r}.'
            )
    for scalar_name in scalar_names:
        uri = str(root / 'scalars' / scalar_name / 'values')
        if tiledb.object_type(uri) != 'array':
            raise RuntimeError(f'Output validation failed: missing TileDB array {uri!r}.')
        with tiledb.open(uri, 'r') as array:
            schema = array.schema
            if schema.sparse or schema.domain.ndim != 2:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    'values must be a two-dimensional dense TileDB array.'
                )
            if schema.nattr != 1 or not schema.has_attr('values'):
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    "array must contain exactly one 'values' attribute."
                )
            if dtype is not None and schema.attr('values').dtype != dtype:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'dtype {schema.attr("values").dtype} != expected {dtype}.'
                )
            element_count = _expected_element_count(expected_elements, scalar_name)
            if element_count is not None and array.shape[1] != element_count:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'{array.shape[1]} elements != expected {element_count}.'
                )
            encoded_names = array.meta.get('column_names')
            try:
                column_names = json.loads(encoded_names) if encoded_names is not None else None
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f'Output validation failed: invalid column_names metadata in {uri!r}.'
                ) from exc
            if not isinstance(column_names, list) or len(column_names) != array.shape[0]:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    f'{array.shape[0]} rows but '
                    f'{len(column_names) if isinstance(column_names, list) else 0} source names.'
                )
            stored_sources = [str(value) for value in column_names]
            if expected_sources is not None:
                expected = [str(value) for value in expected_sources[scalar_name]]
                if stored_sources != expected:
                    raise RuntimeError(
                        f'Output validation failed for scalar {scalar_name!r}: '
                        'source names or order do not match the cohort.'
                    )
            if array.shape[0] and array.shape[1]:
                domain = schema.domain
                first = tuple(int(domain.dim(index).domain[0]) for index in range(2))
                last = tuple(int(domain.dim(index).domain[1]) for index in range(2))
                array[first[0], first[1]]['values']
                array[last[0], last[1]]['values']
        if require_column_name_arrays:
            column_uri = str(root / 'scalars' / scalar_name / 'column_names')
            if tiledb.object_type(column_uri) != 'array':
                raise RuntimeError(
                    f'Output validation failed: missing TileDB array {column_uri!r}.'
                )
            with tiledb.open(column_uri, 'r') as column_array:
                array_sources = _normalized_names(column_array[:]['values'])
            if array_sources != stored_sources:
                raise RuntimeError(
                    f'Output validation failed for scalar {scalar_name!r}: '
                    'column-name array does not match values metadata.'
                )


def path_stats(path: str | Path) -> tuple[int, int]:
    """Return total bytes and regular-file count below a path."""
    output = Path(path)
    if output.is_file():
        return output.stat().st_size, 1
    total_bytes = 0
    file_count = 0
    if output.exists():
        for child in output.rglob('*'):
            if child.is_file():
                file_count += 1
                total_bytes += child.stat().st_size
    return total_bytes, file_count


def scalar_output_stats(
    writer_result,
    scalar_name: str,
    output_path: str | Path,
    *,
    split_output: bool,
) -> tuple[int | None, int | None]:
    """Return non-cumulative bytes/files for one scalar write when measurable."""
    raw = writer_result.get(scalar_name, {}) if isinstance(writer_result, dict) else {}
    if isinstance(raw, dict) and isinstance(raw.get('output_bytes'), int):
        file_count = raw.get('output_file_count')
        return raw['output_bytes'], file_count if isinstance(file_count, int) else None
    if split_output:
        return path_stats(output_path)
    # A combined HDF5 file is still open when per-scalar telemetry is emitted.
    # Its current file size is buffered and cumulative, so report only the final
    # conversion-level size rather than a misleading scalar value.
    return None, None


def peak_rss_bytes() -> int:
    """Return process peak RSS normalized across macOS and Linux."""
    if resource is None:
        return 0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == 'darwin' else rss * 1024


class TimingRecorder:
    """Emit structured INFO timing records and optional JSON Lines output."""

    def __init__(self, jsonl_path: str | Path | None, **context) -> None:
        from modelarrayio.utils.s3_utils import s3_client_pool_connections

        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        context.setdefault('s3_pool_connections', s3_client_pool_connections())
        if context.get('backend') == 'tiledb':
            for key, value in _tiledb_runtime_settings().items():
                context.setdefault(key, value)
        self.context = context
        self.started_wall = time.perf_counter()
        self.started_cpu = time.process_time()
        self.peak_open_file_descriptors = _open_file_descriptor_count()
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **values) -> dict:
        open_descriptors = _open_file_descriptor_count()
        if open_descriptors is not None:
            if self.peak_open_file_descriptors is None:
                self.peak_open_file_descriptors = open_descriptors
            else:
                self.peak_open_file_descriptors = max(
                    self.peak_open_file_descriptors, open_descriptors
                )
        record = {
            'event': event,
            'timestamp_unix': time.time(),
            **self.context,
            **values,
        }
        record.setdefault('open_file_descriptors', open_descriptors)
        record.setdefault('peak_open_file_descriptors_sampled', self.peak_open_file_descriptors)
        logger.info('runtime %s', json.dumps(record, sort_keys=True, default=str))
        if self.jsonl_path is not None:
            try:
                with self.jsonl_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, sort_keys=True, default=str) + '\n')
            except OSError:
                logger.warning(
                    'Could not append runtime metrics to %s; continuing without JSONL output.',
                    self.jsonl_path,
                    exc_info=True,
                )
                self.jsonl_path = None
        return record

    def finish(self, **values) -> dict:
        wall_seconds = time.perf_counter() - self.started_wall
        cpu_seconds = time.process_time() - self.started_cpu
        return self.emit(
            'conversion_complete',
            total_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            cpu_utilization_percent=(100.0 * cpu_seconds / wall_seconds if wall_seconds else 0.0),
            peak_memory_bytes=peak_rss_bytes(),
            **values,
        )


_SCALAR_RECORD_KEYS = frozenset(
    {'scalar_name', 'matrix_shape', 'dtype', 'output_bytes', 'output_file_count'}
)


def _writer_metrics(result: Mapping[str, Mapping[str, object]], scalar_name: str) -> dict:
    """Return a writer's per-scalar metrics minus keys the scalar record sets itself."""
    return {
        key: value
        for key, value in result.get(scalar_name, {}).items()
        if key not in _SCALAR_RECORD_KEYS
    }


def publish_scalar_outputs(
    recorder: TimingRecorder,
    *,
    backend: str,
    output: str | Path,
    split_outputs: bool,
    scalar_names: Sequence[str],
    scalars: LazyScalarMatrices,
    sources_by_scalar: Mapping[str, Sequence[str]],
    storage_dtype: str,
    compression: str,
    compression_level: int | None,
    shuffle: bool,
    chunk_voxels: int,
    target_chunk_mb: float,
    expected_elements: int,
    hdf5_metadata: Callable[[h5py.File], None] | None = None,
    hdf5_required_paths: Sequence[str] = (),
    tiledb_metadata: Callable[[Path], None] | None = None,
    tiledb_required_paths: Sequence[str] = (),
    column_name_arrays: bool = False,
) -> list[Path]:
    """Stage, write, validate and atomically publish every scalar; emit telemetry.

    This is the one per-scalar loop shared by the CIFTI, NIfTI and MIF
    converters. ``scalars`` is indexed once per scalar (each access stages that
    scalar) and the matrix is dropped right after its write. ``hdf5_metadata``
    runs once per HDF5 file right after it is created; ``tiledb_metadata`` runs
    once per TileDB root after its scalar arrays are written. Validation
    happens on the temporary path before publication.
    """
    directory = backend == 'tiledb'
    # Filled in as scalars are accessed, so read it per scalar, not up front.
    load_stats = scalars.stats
    writer_kwargs = {
        'storage_dtype': storage_dtype,
        'compression': compression,
        'compression_level': compression_level,
        'shuffle': shuffle,
        'chunk_voxels': chunk_voxels,
        'target_chunk_mb': target_chunk_mb,
    }

    def _validate(path, names):
        if directory:
            validate_tiledb_output(
                path,
                names,
                require_column_name_arrays=column_name_arrays,
                expected_sources=sources_by_scalar,
                expected_dtype=storage_dtype,
                expected_elements=expected_elements,
                required_paths=tiledb_required_paths,
            )
        else:
            validate_hdf5_output(
                path,
                names,
                expected_sources=sources_by_scalar,
                expected_dtype=storage_dtype,
                expected_elements=expected_elements,
                required_paths=hdf5_required_paths,
            )

    def _write(target, scalar_name):
        started = time.perf_counter()
        matrix = scalars[scalar_name]
        one_scalar = {scalar_name: matrix}
        one_source = {scalar_name: sources_by_scalar[scalar_name]}
        if directory:
            result = cli_utils.write_tiledb_scalar_matrices(
                target,
                one_scalar,
                one_source,
                write_column_name_arrays=column_name_arrays,
                **writer_kwargs,
            )
        else:
            result = cli_utils.write_hdf5_scalar_matrices(
                target, one_scalar, one_source, **writer_kwargs
            )
        return matrix, result, started

    def _emit(scalar_name, matrix, result, started, output_path):
        output_bytes, output_files = scalar_output_stats(
            result, scalar_name, output_path, split_output=split_outputs
        )
        recorder.emit(
            'scalar_complete',
            scalar_name=scalar_name,
            scalar_seconds=time.perf_counter() - started,
            matrix_shape=[int(value) for value in matrix.shape],
            dtype=str(matrix.dtype),
            **load_timing_metrics(load_stats.get(scalar_name)),
            peak_memory_bytes=peak_rss_bytes(),
            output_bytes=output_bytes,
            output_file_count=output_files,
            **_writer_metrics(result, scalar_name),
        )

    published: list[Path] = []
    if split_outputs:
        for scalar_name in scalar_names:
            scalar_output = Path(cli_utils.prefixed_output_path(output, scalar_name))
            with atomic_output(scalar_output, directory=directory) as temporary:
                if directory:
                    matrix, result, started = _write(temporary, scalar_name)
                    if tiledb_metadata is not None:
                        tiledb_metadata(temporary)
                else:
                    with h5py.File(temporary, 'w') as h5_file:
                        if hdf5_metadata is not None:
                            hdf5_metadata(h5_file)
                        matrix, result, started = _write(h5_file, scalar_name)
                _validate(temporary, [scalar_name])
                _emit(scalar_name, matrix, result, started, temporary)
                del matrix
            published.append(scalar_output)
    else:
        final_output = Path(output)
        with atomic_output(final_output, directory=directory) as temporary:
            if directory:
                for scalar_name in scalar_names:
                    matrix, result, started = _write(temporary, scalar_name)
                    _emit(scalar_name, matrix, result, started, temporary)
                    del matrix
                if tiledb_metadata is not None:
                    tiledb_metadata(temporary)
            else:
                with h5py.File(temporary, 'w') as h5_file:
                    if hdf5_metadata is not None:
                        hdf5_metadata(h5_file)
                    for scalar_name in scalar_names:
                        matrix, result, started = _write(h5_file, scalar_name)
                        _emit(scalar_name, matrix, result, started, temporary)
                        del matrix
            _validate(temporary, list(scalar_names))
        published.append(final_output)

    output_stats = [path_stats(path) for path in published]
    recorder.finish(
        scalar_count=len(scalar_names),
        output_bytes=sum(total_bytes for total_bytes, _ in output_stats),
        output_file_count=sum(file_count for _, file_count in output_stats),
    )
    return published
