"""Bounded, ordered source loading into typed staging matrices."""

from __future__ import annotations

import logging
import operator
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

SourceT = TypeVar('SourceT')

# Seconds between progress lines while a scalar is being staged.
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0

S3_WORKERS_DEPRECATION_MESSAGE = (
    '--s3-workers is deprecated and will be removed in the next breaking release; '
    'use --workers instead. --workers now controls source loading for local and S3 input.'
)


class SourceLoadError(RuntimeError):
    """A source could not be loaded into its staging-matrix row."""


@dataclass(frozen=True)
class LoadedRow:
    """A decoded row plus optional validation time recorded by its reader."""

    values: np.ndarray
    validation_seconds: float = 0.0
    source_seconds: float = 0.0
    cpu_seconds: float = 0.0
    # True when a reader took a fast path (for example the CIFTI raw data-block
    # read) instead of a full decode, so silent fallbacks are visible in telemetry.
    fast_path: bool = False


@dataclass
class LoadStats:
    """Cheap aggregate counters for one scalar's bounded load."""

    allocation_seconds: float = 0.0
    wall_seconds: float = 0.0
    source_wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    source_seconds: float = 0.0
    validation_seconds: float = 0.0
    scheduler_wait_seconds: float = 0.0
    in_flight_sources_mean: float = 0.0
    in_flight_sources_max: int = 0
    # Rows whose reader reported a fast path (see LoadedRow.fast_path). The
    # first row of a scalar is its reference decode, so a fully served scalar
    # reports ``source_count - 1`` here.
    fast_path_rows: int = 0
    source_count: int = 0
    workers: int = 1

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable counters."""
        return asdict(self)


def validate_workers(workers: int | None) -> int:
    """Return a positive loader-worker count."""
    if workers is None:
        return 1
    try:
        if isinstance(workers, bool):
            raise TypeError('a boolean is not a worker count')  # bool passes operator.index
        resolved = operator.index(workers)
    except TypeError as exc:
        raise ValueError(f'workers must be a positive integer, got {workers!r}') from exc
    if resolved < 1:
        raise ValueError(f'workers must be a positive integer, got {workers!r}')
    return resolved


def resolve_loader_workers(workers: int | None, s3_workers: int | None) -> int:
    """Resolve ``workers`` and its deprecated ``s3_workers`` alias to one count.

    This is the single place the alias is interpreted, for the CLI
    (``--workers`` / ``--s3-workers``) and for the modality loaders alike.
    """
    if workers is None and s3_workers is None:
        return 1

    resolved_workers = validate_workers(workers) if workers is not None else None
    resolved_s3 = validate_workers(s3_workers) if s3_workers is not None else None
    if resolved_s3 is not None:
        # FutureWarning (not DeprecationWarning) because the audience is CLI
        # users: Python shows it by default, so one emission is enough.
        warnings.warn(S3_WORKERS_DEPRECATION_MESSAGE, FutureWarning, stacklevel=3)
    if (
        resolved_workers is not None
        and resolved_s3 is not None
        and resolved_workers != resolved_s3
    ):
        raise ValueError(
            f'Conflicting loader worker counts: --workers={resolved_workers} and '
            f'--s3-workers={resolved_s3}. Use one option or pass matching values.'
        )
    return resolved_workers if resolved_workers is not None else int(resolved_s3)


def _source_label(source: Any) -> str:
    if isinstance(source, Mapping):
        for key in ('source_file', 'path', 'uri'):
            if key in source:
                return str(source[key])
    if isinstance(source, tuple) and source:
        return str(source[0])
    return str(source)


def _unwrap_loaded_row(
    value: np.ndarray | LoadedRow,
) -> tuple[np.ndarray, float, float, float, bool]:
    if isinstance(value, LoadedRow):
        return (
            np.asarray(value.values),
            float(value.validation_seconds),
            float(value.source_seconds),
            float(value.cpu_seconds),
            bool(value.fast_path),
        )
    return np.asarray(value), 0.0, 0.0, 0.0, False


def stage_source_rows(
    sources: Sequence[SourceT],
    load_row: Callable[[SourceT], np.ndarray | LoadedRow],
    *,
    scalar_name: str,
    storage_dtype: str | np.dtype | type[np.floating] = np.float32,
    workers: int = 1,
    n_elements: int | None = None,
    first_row: np.ndarray | LoadedRow | None = None,
    progress_interval_seconds: float | None = DEFAULT_PROGRESS_INTERVAL_SECONDS,
) -> tuple[np.ndarray, LoadStats]:
    """Load ordered sources into one C-contiguous staging matrix.

    Only ``workers`` calls are submitted or running at once. Each task copies
    directly into its assigned row, so completion order never changes cohort
    order and no second list/stack copy is built. While the load runs, an INFO
    progress line is logged at most every ``progress_interval_seconds`` (``None``
    or a non-positive value disables it), so a long scalar is never silent.
    """
    worker_count = validate_workers(workers)
    source_count = len(sources)
    if source_count == 0:
        raise ValueError(f'Cannot stage scalar {scalar_name!r}: no sources were provided.')

    dtype = np.dtype(storage_dtype)
    stats = LoadStats(source_count=source_count, workers=worker_count)
    started = time.perf_counter()
    started_cpu = time.process_time()

    loaded_first: np.ndarray | None = None
    start_index = 0
    preloaded_source_seconds = 0.0
    preloaded_cpu_seconds = 0.0
    in_flight_total = 0
    in_flight_samples = 0
    in_flight_max = 0

    def _sample_in_flight(count: int) -> None:
        nonlocal in_flight_total, in_flight_samples, in_flight_max
        if count <= 0:
            return
        in_flight_total += count
        in_flight_samples += 1
        in_flight_max = max(in_flight_max, count)

    if first_row is not None:
        (
            loaded_first,
            validation_seconds,
            preloaded_source_seconds,
            preloaded_cpu_seconds,
            first_fast_path,
        ) = _unwrap_loaded_row(first_row)
        stats.validation_seconds += validation_seconds
        stats.fast_path_rows += int(first_fast_path)
        stats.source_seconds += preloaded_source_seconds
        stats.source_wall_seconds += preloaded_source_seconds
        if preloaded_source_seconds:
            _sample_in_flight(1)
        start_index = 1
    elif n_elements is None:
        source_started = time.perf_counter()
        try:
            loaded_first, validation_seconds, _, _, first_fast_path = _unwrap_loaded_row(
                load_row(sources[0])
            )
        except Exception as exc:
            raise SourceLoadError(
                f'Failed to load source {_source_label(sources[0])!r} '
                f'for scalar {scalar_name!r}: {exc}'
            ) from exc
        first_source_seconds = time.perf_counter() - source_started
        stats.source_seconds += first_source_seconds
        stats.source_wall_seconds += first_source_seconds
        stats.validation_seconds += validation_seconds
        stats.fast_path_rows += int(first_fast_path)
        _sample_in_flight(1)
        start_index = 1

    if loaded_first is not None:
        if loaded_first.ndim != 1:
            raise SourceLoadError(
                f'Loaded source {_source_label(sources[0])!r} for scalar {scalar_name!r} '
                f'produced shape {loaded_first.shape}; expected one dimension.'
            )
        inferred_elements = int(loaded_first.shape[0])
        if n_elements is None:
            n_elements = inferred_elements
        elif inferred_elements != int(n_elements):
            raise SourceLoadError(
                f'Loaded source {_source_label(sources[0])!r} for scalar {scalar_name!r} '
                f'has {inferred_elements} elements; expected {n_elements}.'
            )

    if n_elements is None or int(n_elements) <= 0:
        raise ValueError(
            f'Cannot stage scalar {scalar_name!r}: n_elements must be positive, got {n_elements!r}.'
        )
    n_elements = int(n_elements)

    allocation_started = time.perf_counter()
    matrix = np.empty((source_count, n_elements), dtype=dtype, order='C')
    stats.allocation_seconds = time.perf_counter() - allocation_started
    if loaded_first is not None:
        np.copyto(matrix[0], loaded_first, casting='unsafe')
        del loaded_first

    def _load_into(index: int, source: SourceT) -> tuple[float, float, bool]:
        row_started = time.perf_counter()
        try:
            values, validation_seconds, _, _, fast_path = _unwrap_loaded_row(load_row(source))
            if values.ndim != 1 or values.shape[0] != n_elements:
                raise ValueError(f'loaded row has shape {values.shape}; expected ({n_elements},)')
            np.copyto(matrix[index], values, casting='unsafe')
        except Exception as exc:
            raise SourceLoadError(
                f'Failed to load source {_source_label(source)!r} '
                f'for scalar {scalar_name!r}: {exc}'
            ) from exc
        return time.perf_counter() - row_started, validation_seconds, fast_path

    progress_interval = (
        float(progress_interval_seconds) if progress_interval_seconds is not None else 0.0
    )
    progress_started = time.perf_counter()
    last_progress = progress_started
    completed = start_index

    def _record_completion(in_flight: int) -> None:
        nonlocal completed, last_progress
        completed += 1
        if progress_interval <= 0 or completed >= source_count:
            return
        now = time.perf_counter()
        if now - last_progress < progress_interval:
            return
        last_progress = now
        elapsed = now - progress_started
        logger.info(
            'Staging %s: %d/%d sources loaded (%.1f sources/s, %d in flight)',
            scalar_name,
            completed,
            source_count,
            completed / elapsed if elapsed > 0 else 0.0,
            in_flight,
        )

    remaining = range(start_index, source_count)
    remaining_started = time.perf_counter()
    loaded_remaining = False
    if worker_count == 1:
        for index in remaining:
            loaded_remaining = True
            _sample_in_flight(1)
            elapsed, validation_seconds, fast_path = _load_into(index, sources[index])
            stats.source_seconds += elapsed
            stats.validation_seconds += validation_seconds
            stats.fast_path_rows += int(fast_path)
            _record_completion(0)
    else:
        source_iter: Iterator[int] = iter(remaining)
        futures: dict[Future[tuple[float, float, bool]], int] = {}
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix='modelarray-loader',
        ) as executor:
            for _ in range(worker_count):
                try:
                    index = next(source_iter)
                except StopIteration:
                    break
                futures[executor.submit(_load_into, index, sources[index])] = index

            try:
                while futures:
                    loaded_remaining = True
                    _sample_in_flight(len(futures))
                    wait_started = time.perf_counter()
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    stats.scheduler_wait_seconds += time.perf_counter() - wait_started
                    for future in done:
                        futures.pop(future)
                        elapsed, validation_seconds, fast_path = future.result()
                        stats.source_seconds += elapsed
                        stats.validation_seconds += validation_seconds
                        stats.fast_path_rows += int(fast_path)
                        try:
                            index = next(source_iter)
                        except StopIteration:
                            pass
                        else:
                            futures[executor.submit(_load_into, index, sources[index])] = index
                        _record_completion(len(futures))
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    if loaded_remaining:
        stats.source_wall_seconds += time.perf_counter() - remaining_started
    if in_flight_samples:
        stats.in_flight_sources_mean = in_flight_total / in_flight_samples
        stats.in_flight_sources_max = in_flight_max
    stats.wall_seconds = time.perf_counter() - started
    if preloaded_source_seconds:
        # The preloaded row was decoded before this function could infer and
        # allocate the matrix. Include it in end-to-end staging telemetry.
        stats.wall_seconds += preloaded_source_seconds
    stats.cpu_seconds = time.process_time() - started_cpu + preloaded_cpu_seconds
    return matrix, stats


class LazyScalarMatrices(Mapping[str, np.ndarray], Generic[SourceT]):
    """Mapping that stages one scalar matrix at a time on access.

    Every access re-runs the staging function: there is no cache, so index each
    scalar once, write it, and drop it. Loader-specific setup products travel
    with the mapping as explicit attributes instead of being attached ad hoc:

    ``stats``
        Per-scalar :class:`LoadStats`, filled in as scalars are accessed.
    ``sources_by_scalar``
        Ordered source paths per scalar, exactly as the rows were staged.
    ``reference_image``, ``reference_brain_names``, ``reference_signature``
        CIFTI only: the decoded first image, its element names and spatial
        signature, so callers can build metadata without reopening the file.
    ``flat_indices``, ``mask_cache``
        NIfTI only: the group-mask flat selection and the reduced-mask cache.
    """

    def __init__(
        self,
        scalar_names: Sequence[str],
        stage_scalar: Callable[[str], tuple[np.ndarray, LoadStats]],
        *,
        sources_by_scalar: Mapping[str, Sequence[Any]] | None = None,
        reference_image: Any = None,
        reference_brain_names: np.ndarray | None = None,
        reference_signature: tuple[int, str] | None = None,
        flat_indices: np.ndarray | None = None,
        mask_cache: Any = None,
    ) -> None:
        self._scalar_names = tuple(scalar_names)
        self._stage_scalar = stage_scalar
        self.stats: dict[str, LoadStats] = {}
        self.sources_by_scalar = sources_by_scalar
        self.reference_image = reference_image
        self.reference_brain_names = reference_brain_names
        self.reference_signature = reference_signature
        self.flat_indices = flat_indices
        self.mask_cache = mask_cache

    def __getitem__(self, scalar_name: str) -> np.ndarray:
        if scalar_name not in self._scalar_names:
            raise KeyError(scalar_name)
        matrix, stats = self._stage_scalar(scalar_name)
        self.stats[scalar_name] = stats
        return matrix

    def __iter__(self) -> Iterator[str]:
        return iter(self._scalar_names)

    def __len__(self) -> int:
        return len(self._scalar_names)
