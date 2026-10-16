"""HDF5 storage utilities."""

from __future__ import annotations

import logging
import operator
import os
import time
import zlib
from collections.abc import MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, product

import h5py
import numpy as np
import pandas as pd

from modelarrayio.storage import utils as storage_utils

logger = logging.getLogger(__name__)

_H5_FILTER_DEFLATE = 1
_H5_FILTER_SHUFFLE = 2


class _DirectChunkUnsupported(RuntimeError):
    """Raised when a dataset's filter pipeline cannot be reproduced safely."""


def _write_column_names_to_group(
    group: h5py.Group,
    sources: pd.Series | Sequence[str],
) -> h5py.Dataset:
    """Write source names beside the values dataset in *group*."""
    values = np.array(storage_utils.normalize_column_names(sources), dtype=object)
    return group.create_dataset(
        'column_names',
        data=values,
        dtype=h5py.string_dtype(encoding='utf-8'),
        shape=(len(values),),
    )


def resolve_dtype(storage_dtype):
    """Resolve a storage dtype to a supported NumPy floating type.

    Parameters
    ----------
    storage_dtype : :obj:`str`
        Storage dtype.

    Returns
    -------
    :obj:`numpy.dtype`
        Supported NumPy floating type.
    """
    return storage_utils.resolve_dtype(storage_dtype)


def resolve_compression(compression, compression_level, shuffle):
    """Resolve a compression method to a supported compression method.

    Parameters
    ----------
    compression : :obj:`str`
        Compression method.
    compression_level : :obj:`int`
        Compression level.
    shuffle : :obj:`bool`
        Whether to shuffle the data.

    Returns
    -------
    :obj:`tuple`
        Compression method, compression level, and whether to shuffle the data.
    """
    comp = (
        None
        if compression is None or str(compression).lower() == 'none'
        else str(compression).lower()
    )
    use_shuffle = bool(shuffle) if comp is not None else False
    gzip_level = None
    if comp == 'gzip':
        try:
            gzip_level = int(compression_level)
        except (TypeError, ValueError):
            gzip_level = 4
        gzip_level = max(0, min(9, gzip_level))
    return comp, gzip_level, use_shuffle


def compute_chunk_shape_full_subjects(
    n_files, n_elements, item_chunk, target_chunk_mb, storage_np_dtype
):
    """Compute a chunk shape for a full subject.

    Parameters
    ----------
    n_files : :obj:`int`
        Number of subjects.
    n_elements : :obj:`int`
        Number of items.
    item_chunk : :obj:`int`
        Item chunk.
    target_chunk_mb : :obj:`float`
        Target chunk size in MB.
    storage_np_dtype : :obj:`numpy.dtype`
        Storage numpy dtype.

    Returns
    -------
    :obj:`tuple`
        Chunk shape.
    """
    chunk = storage_utils.compute_full_subject_chunk_shape(
        n_files=n_files,
        n_elements=n_elements,
        item_chunk=item_chunk,
        target_chunk_mb=target_chunk_mb,
        storage_np_dtype=storage_np_dtype,
    )
    logger.debug(
        'Computed chunk shape: %s (subjects=%d, items=%d, item_chunk=%s, target_chunk_mb=%.2f)',
        chunk,
        n_files,
        n_elements,
        str(item_chunk),
        float(target_chunk_mb),
    )
    return chunk


def create_scalar_matrix_dataset(
    h5file,
    dataset_path,
    stacked_values,
    sources_list,
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    metrics: MutableMapping[str, object] | None = None,
):
    """Create a scalar matrix dataset in an HDF5 file.

    Parameters
    ----------
    h5file : :obj:`h5py.File`
        HDF5 file.
    dataset_path : :obj:`str`
        Dataset path.
    stacked_values : :obj:`numpy.ndarray`
        Stacked values.
    sources_list : :obj:`list`
        Sources list.
    storage_dtype : :obj:`str`
        Storage dtype.
    compression : :obj:`str`
        Compression method.
    compression_level : :obj:`int`
        Compression level.
    shuffle : :obj:`bool`
        Whether to shuffle the data.
    chunk_voxels : :obj:`int`
        Chunk voxels.
    target_chunk_mb : :obj:`float`
        Target chunk size in MB.

    Returns
    -------
    :obj:`h5py.Dataset`
        Scalar matrix dataset.
    """
    storage_np_dtype = resolve_dtype(storage_dtype)
    comp, comp_opts, use_shuffle = resolve_compression(compression, compression_level, shuffle)

    stacked_values = _coerce_full_matrix(stacked_values, dtype=storage_np_dtype)

    n_files, n_elements = stacked_values.shape
    chunk_shape = compute_chunk_shape_full_subjects(
        n_files, n_elements, chunk_voxels, target_chunk_mb, storage_np_dtype
    )
    logger.info(
        'Creating dataset %s with shape (%d, %d), dtype=%s, chunks=%s, compression=%s',
        dataset_path,
        n_files,
        n_elements,
        storage_np_dtype,
        chunk_shape,
        str(comp),
    )
    dset = h5file.create_dataset(
        dataset_path,
        shape=(n_files, n_elements),
        dtype=storage_np_dtype,
        chunks=chunk_shape,
        compression=comp,
        compression_opts=comp_opts if comp == 'gzip' else None,
        shuffle=use_shuffle,
    )
    logger.info('Writing full dataset %s to HDF5 (this may take a while)...', dataset_path)
    write_metrics = write_full_matrix(dset, stacked_values)
    logger.info('Finished writing dataset %s', dataset_path)
    if sources_list is not None:
        _write_column_names_to_group(dset.parent, sources_list)
    if metrics is not None:
        metrics.update(write_metrics)
        metrics['dataset_path'] = str(dataset_path)
    return dset


def create_empty_scalar_matrix_dataset(
    h5file,
    dataset_path,
    n_files,
    n_elements,
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    sources_list: pd.Series | list | None = None,
):
    """Create an empty scalar matrix dataset in an HDF5 file.

    Parameters
    ----------
    h5file : :obj:`h5py.File`
        HDF5 file.
    dataset_path : :obj:`str`
        Dataset path.
    n_files : :obj:`int`
        Number of subjects.
    n_elements : :obj:`int`
        Number of items.
    storage_dtype : :obj:`str`
        Storage dtype.
    compression : :obj:`str`
        Compression method.
    compression_level : :obj:`int`
        Compression level.
    shuffle : :obj:`bool`
        Whether to shuffle the data.
    chunk_voxels : :obj:`int`
        Chunk voxels.
    target_chunk_mb : :obj:`float`
        Target chunk size in MB.
    sources_list : :obj:`list`
        Sources list.

    Returns
    -------
    :obj:`h5py.Dataset`
        Empty scalar matrix dataset.
    """
    storage_np_dtype = resolve_dtype(storage_dtype)
    comp, comp_opts, use_shuffle = resolve_compression(compression, compression_level, shuffle)

    chunk_shape = compute_chunk_shape_full_subjects(
        n_files, n_elements, chunk_voxels, target_chunk_mb, storage_np_dtype
    )
    logger.info(
        'Creating empty dataset %s with shape (%d, %d), dtype=%s, chunks=%s, compression=%s',
        dataset_path,
        n_files,
        n_elements,
        storage_np_dtype,
        chunk_shape,
        str(comp),
    )
    dset = h5file.create_dataset(
        dataset_path,
        shape=(n_files, n_elements),
        dtype=storage_np_dtype,
        chunks=chunk_shape,
        compression=comp,
        compression_opts=comp_opts if comp == 'gzip' else None,
        shuffle=use_shuffle,
    )
    if sources_list is not None:
        _write_column_names_to_group(dset.parent, sources_list)
    return dset


def write_column_names(h5_file: h5py.File, scalar: str, sources: pd.Series | list):
    """Write column names to an HDF5 file.

    Parameters
    ----------
    h5_file : :obj:`h5py.File`
        HDF5 file.
    scalar : :obj:`str`
        Scalar name.
    sources : :obj:`list`
        Sources list.
    """
    scalar = storage_utils.validate_scalar_name(scalar)
    grp = h5_file.require_group(f'scalars/{scalar}')
    _write_column_names_to_group(grp, sources)


# Shared with the other backend; kept under the private name for callers and tests.
_coerce_full_matrix = storage_utils.coerce_full_matrix


def _hdf5_byte_shuffle(raw: bytes, element_size: int) -> bytes:
    """Apply the HDF5 shuffle filter's byte transpose exactly."""
    if element_size <= 0:
        raise ValueError('element_size must be positive')
    byte_values = np.frombuffer(raw, dtype=np.uint8)
    if byte_values.size % element_size:
        raise ValueError('raw chunk size must be divisible by element_size')
    return byte_values.reshape(-1, element_size).T.tobytes(order='C')


def _chunk_offsets(shape: tuple[int, int], chunk_shape: tuple[int, int]) -> list[tuple[int, int]]:
    return list(
        product(
            range(0, shape[0], chunk_shape[0]),
            range(0, shape[1], chunk_shape[1]),
        )
    )


def _encode_gzip_chunk(
    matrix: np.ndarray,
    offset: tuple[int, int],
    chunk_shape: tuple[int, int],
    fill_value: object,
    compression_level: int,
    shuffle: bool,
) -> tuple[tuple[int, int], bytes, float, float]:
    """Pad, optionally byte-shuffle, and gzip one HDF5 chunk."""
    wall_started = time.perf_counter()
    cpu_started = time.thread_time()
    row_start, column_start = offset
    row_stop = min(row_start + chunk_shape[0], matrix.shape[0])
    column_stop = min(column_start + chunk_shape[1], matrix.shape[1])
    source = matrix[row_start:row_stop, column_start:column_stop]

    if source.shape == chunk_shape:
        chunk = np.ascontiguousarray(source)
    else:
        chunk = np.empty(chunk_shape, dtype=matrix.dtype, order='C')
        chunk.fill(fill_value)
        chunk[: source.shape[0], : source.shape[1]] = source

    raw = chunk.tobytes(order='C')
    if shuffle:
        raw = _hdf5_byte_shuffle(raw, matrix.dtype.itemsize)
    compressed = zlib.compress(raw, level=compression_level)
    return (
        offset,
        compressed,
        time.perf_counter() - wall_started,
        time.thread_time() - cpu_started,
    )


def _available_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity/cgroup limits."""
    process_cpu_count = getattr(os, 'process_cpu_count', None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return max(1, int(count))

    sched_getaffinity = getattr(os, 'sched_getaffinity', None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, os.cpu_count() or 1)


def _resolve_compression_workers(compression_workers: int | None) -> int:
    """Resolve and validate the direct-chunk compression worker count."""
    configured_workers: object = compression_workers
    if configured_workers is None:
        configured_workers = os.environ.get('MODELARRAYIO_HDF5_COMPRESSION_WORKERS')
        if configured_workers is None:
            return _available_cpu_count()
        try:
            requested_workers = int(configured_workers)
        except ValueError as exc:
            raise ValueError(
                'MODELARRAYIO_HDF5_COMPRESSION_WORKERS must be a positive integer'
            ) from exc
    else:
        if isinstance(configured_workers, bool):
            raise ValueError('HDF5 compression workers must be a positive integer')
        try:
            requested_workers = operator.index(configured_workers)
        except TypeError as exc:
            raise ValueError('HDF5 compression workers must be a positive integer') from exc

    if requested_workers < 1:
        raise ValueError('HDF5 compression workers must be a positive integer')
    return requested_workers


def _validate_direct_chunk_pipeline(dset: h5py.Dataset) -> None:
    """Verify that direct bytes can reproduce the dataset's full filter pipeline."""
    if dset.chunks is None:
        raise _DirectChunkUnsupported('the dataset is not chunked')
    if dset.compression != 'gzip':
        raise _DirectChunkUnsupported(
            f'compression {dset.compression!r} is not supported by the direct-chunk writer'
        )
    if not hasattr(dset.id, 'write_direct_chunk'):
        raise _DirectChunkUnsupported('h5py does not expose write_direct_chunk')

    expected: list[tuple[int, tuple[int, ...]]] = []
    if dset.shuffle:
        expected.append((_H5_FILTER_SHUFFLE, (dset.dtype.itemsize,)))
    expected.append((_H5_FILTER_DEFLATE, (int(dset.compression_opts),)))

    creation_properties = dset.id.get_create_plist()
    actual = []
    for index in range(creation_properties.get_nfilters()):
        filter_id, _flags, client_data, _name = creation_properties.get_filter(index)
        actual.append((int(filter_id), tuple(int(value) for value in client_data)))
    if actual != expected:
        raise _DirectChunkUnsupported(
            f'the declared HDF5 filter pipeline {actual!r} does not match {expected!r}'
        )


def _write_direct_gzip_chunks(
    dset: h5py.Dataset,
    matrix: np.ndarray,
    *,
    compression_workers: int | None,
) -> dict[str, object]:
    """Compress chunks in worker threads and submit bytes from the calling thread."""
    _validate_direct_chunk_pipeline(dset)
    chunk_shape = tuple(int(value) for value in dset.chunks)
    offsets = _chunk_offsets(matrix.shape, chunk_shape)
    requested_workers = _resolve_compression_workers(compression_workers)
    worker_count = max(1, min(requested_workers, len(offsets)))
    compression_seconds = 0.0
    compression_cpu_seconds = 0.0
    submission_seconds = 0.0

    offset_iterator = iter(offsets)
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix='modelarrayio-hdf5',
    ) as executor:
        while batch := tuple(islice(offset_iterator, worker_count)):
            compression_started = time.perf_counter()
            futures = [
                executor.submit(
                    _encode_gzip_chunk,
                    matrix,
                    offset,
                    chunk_shape,
                    dset.fillvalue,
                    int(dset.compression_opts),
                    bool(dset.shuffle),
                )
                for offset in batch
            ]
            encoded_chunks = [future.result() for future in futures]
            compression_seconds += time.perf_counter() - compression_started
            compression_cpu_seconds += sum(result[3] for result in encoded_chunks)

            for offset, compressed, _worker_wall_seconds, _worker_cpu_seconds in encoded_chunks:
                submission_started = time.perf_counter()
                dset.id.write_direct_chunk(offset, compressed, filter_mask=0)
                submission_seconds += time.perf_counter() - submission_started

    return {
        'compression_seconds': compression_seconds,
        'compression_cpu_seconds': compression_cpu_seconds,
        'submission_seconds': submission_seconds,
        'compression_workers': worker_count,
        'chunk_count': len(offsets),
    }


def write_full_matrix(
    dset: h5py.Dataset,
    values: np.ndarray | Sequence[np.ndarray],
    *,
    compression_workers: int | None = None,
) -> dict[str, object]:
    """Write a complete matrix, using parallel direct-chunk gzip when safe.

    The HDF5 handle is only accessed by the calling thread. Worker threads receive
    NumPy slices and return fully filtered byte buffers. Unsupported filters and
    worker failures are logged and safely rewritten through ordinary h5py assignment.
    """
    matrix = _coerce_full_matrix(values, dtype=dset.dtype, shape=dset.shape)
    started = time.perf_counter()
    metrics: dict[str, object] = {
        'matrix_shape': [int(value) for value in matrix.shape],
        'dtype': matrix.dtype.name,
        'chunk_shape': [int(value) for value in dset.chunks] if dset.chunks else None,
        'compression': dset.compression or 'none',
        'compression_level': (int(dset.compression_opts) if dset.compression == 'gzip' else None),
        'shuffle': bool(dset.shuffle),
        'direct_chunk': False,
        'fallback': False,
        'fallback_reason': None,
        'compression_seconds': 0.0,
        'compression_cpu_seconds': 0.0,
        'submission_seconds': 0.0,
        'compression_workers': 0,
        'chunk_count': 0,
    }

    if dset.compression is None:
        submission_started = time.perf_counter()
        dset[...] = matrix
        metrics['submission_seconds'] = time.perf_counter() - submission_started
    else:
        # Configuration errors are user errors, not direct-writer failures. Resolve
        # them before entering the fallback boundary so they cannot be hidden by a
        # successful but unexpectedly serial h5py write.
        resolved_workers = _resolve_compression_workers(compression_workers)
        try:
            direct_metrics = _write_direct_gzip_chunks(
                dset,
                matrix,
                compression_workers=resolved_workers,
            )
        # Any worker or filter failure must fall back safely; the debug log keeps the traceback.
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            logger.warning(
                'Falling back to ordinary HDF5 dataset assignment for %s: %s',
                dset.name,
                reason,
            )
            logger.debug('Direct-chunk write failure details', exc_info=True)
            submission_started = time.perf_counter()
            dset[...] = matrix
            metrics['submission_seconds'] = time.perf_counter() - submission_started
            metrics['compression_seconds'] = None
            metrics['compression_cpu_seconds'] = None
            metrics['fallback'] = True
            metrics['fallback_reason'] = reason
        else:
            metrics.update(direct_metrics)
            metrics['direct_chunk'] = True

    metrics['write_seconds'] = time.perf_counter() - started
    return metrics


def write_rows_in_column_stripes(dset, rows):
    """Compatibility wrapper that now writes rows as one full matrix.

    Parameters
    ----------
    dset : h5py.Dataset
        Target dataset with shape (n_files, n_elements) and chunking set.
    rows : Sequence[np.ndarray]
        List/sequence of 1D arrays, one per subject, length == n_elements.
        Each will be cast into one C-contiguous staging matrix if needed.

    Returns
    -------
    dict
        JSON-serializable write metrics from :func:`write_full_matrix`.
    """
    return write_full_matrix(dset, rows)
