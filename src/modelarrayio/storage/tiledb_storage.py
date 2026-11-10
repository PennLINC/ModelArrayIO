"""TileDB storage utilities."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import MutableMapping, Sequence

import numpy as np
import tiledb

from modelarrayio.storage import utils as storage_utils

logger = logging.getLogger(__name__)

# TileDBArray R package expects dimensions to be R integers.
_DIM_DTYPE = np.int32


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


def _resolve_compression_settings(
    compression: str | None,
    compression_level: int | None,
) -> tuple[str, int | None]:
    """Return the effective TileDB codec and level used by the schema.

    ``None`` means "no compression" at this storage level; backend defaults are
    chosen by the CLI before it reaches here. Unsupported codecs and invalid
    levels raise ``ValueError``, exactly as
    :func:`modelarrayio.storage.utils.resolve_backend_compression` does.
    """
    return storage_utils.resolve_backend_compression(
        'tiledb', 'none' if compression is None else compression, compression_level
    )


def _build_filter_list(compression: str | None, compression_level: int | None, shuffle: bool):
    compression, compression_level = _resolve_compression_settings(compression, compression_level)
    filters = []
    # A standalone byte-shuffle still consumes CPU even though it cannot
    # reduce output size. Keep ``--compression none`` as the true no-filter
    # path, matching HDF5 and the CLI description.
    if shuffle and compression != 'none':
        # ByteShuffle works well for float data; BitShuffle is also available
        filters.append(tiledb.ByteShuffleFilter())
    if compression == 'zstd':
        filters.append(tiledb.ZstdFilter(level=compression_level))
    elif compression == 'gzip':
        filters.append(tiledb.GzipFilter(level=compression_level))
    return tiledb.FilterList(filters)


def compute_tile_shape_full_subjects(
    n_files, n_elements, item_tile, target_tile_mb, storage_np_dtype
):
    """Compute a tile shape for a full subject.

    Parameters
    ----------
    n_files : :obj:`int`
        Number of subjects.
    n_elements : :obj:`int`
        Number of items.
    item_tile : :obj:`int`
        Item tile.
    target_tile_mb : :obj:`float`
        Target tile size in MB.
    storage_np_dtype : :obj:`numpy.dtype`
        Storage numpy dtype.

    Returns
    -------
    :obj:`tuple`
        Tile shape.
    """
    tile = storage_utils.compute_full_subject_chunk_shape(
        n_files=n_files,
        n_elements=n_elements,
        item_chunk=item_tile,
        target_chunk_mb=target_tile_mb,
        storage_np_dtype=storage_np_dtype,
    )
    logger.debug(
        'Computed tile shape: %s (subjects=%d, items=%d, item_tile=%s, target_tile_mb=%.2f)',
        tile,
        n_files,
        n_elements,
        str(item_tile),
        float(target_tile_mb),
    )
    return tile


def _ensure_parent_group(uri: str):
    parent = os.path.dirname(uri.rstrip('/'))
    if not parent or parent == uri:
        return
    if not tiledb.object_type(parent):
        if not os.path.exists(parent):
            _ensure_parent_group(parent)
        tiledb.group_create(parent)


def create_scalar_matrix_array(
    base_uri,
    dataset_path,
    stacked_values,
    sources_list,
    storage_dtype='float32',
    compression='zstd',
    compression_level=5,
    shuffle=True,
    tile_voxels=0,
    target_tile_mb=2.0,
    metrics: MutableMapping[str, object] | None = None,
):
    """Create a scalar matrix array in a TileDB directory.

    Parameters
    ----------
    base_uri : :obj:`str`
        Base URI.
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
    tile_voxels : :obj:`int`
        Tile voxels.
    target_tile_mb : :obj:`float`
        Target tile size in MB.

    Returns
    -------
    :obj:`str`
        URI of the created array.
    """
    storage_np_dtype = resolve_dtype(storage_dtype)
    stacked_values = _coerce_full_matrix(stacked_values, dtype=storage_np_dtype)

    n_files, n_elements = stacked_values.shape
    tile_shape = compute_tile_shape_full_subjects(
        n_files, n_elements, tile_voxels, target_tile_mb, storage_np_dtype
    )

    uri = os.path.join(base_uri, dataset_path)
    _ensure_parent_group(uri)

    # Domain and schema
    dim_subjects = tiledb.Dim(
        name='subjects', domain=(0, n_files - 1), tile=tile_shape[0], dtype=_DIM_DTYPE
    )
    dim_items = tiledb.Dim(
        name='items', domain=(0, n_elements - 1), tile=tile_shape[1], dtype=_DIM_DTYPE
    )
    dom = tiledb.Domain(dim_subjects, dim_items)
    effective_compression, effective_level = _resolve_compression_settings(
        compression, compression_level
    )
    effective_shuffle = bool(shuffle and effective_compression != 'none')
    attr_filters = _build_filter_list(effective_compression, effective_level, shuffle)
    attr_values = tiledb.Attr(name='values', dtype=storage_np_dtype, filters=attr_filters)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    logger.info(
        'Creating TileDB array %s with shape (%d, %d), dtype=%s, tiles=%s',
        uri,
        n_files,
        n_elements,
        storage_np_dtype,
        tile_shape,
    )
    if tiledb.object_type(uri):
        logger.warning('Removing existing array %s', uri)
        tiledb.remove(uri)
    tiledb.Array.create(uri, schema)

    logger.info('Writing full array %s to TileDB in one dense submission...', uri)
    write_started = time.perf_counter()
    with tiledb.open(uri, 'w') as A:
        submission_started = time.perf_counter()
        A[:] = {'values': stacked_values}
        submission_seconds = time.perf_counter() - submission_started
        if sources_list is not None:
            try:
                A.meta['column_names'] = json.dumps(
                    storage_utils.normalize_column_names(sources_list)
                )
            except (TypeError, ValueError, tiledb.TileDBError):
                # Fallback without metadata if serialization fails
                logger.warning('Failed to write column_names metadata for %s', uri)
    write_seconds = time.perf_counter() - write_started
    if metrics is not None:
        fragment_count = len(tiledb.array_fragments(uri))
        output_file_count, output_bytes = _local_output_stats(uri)
        metrics.update(
            {
                'dataset_path': str(dataset_path),
                'matrix_shape': [int(value) for value in stacked_values.shape],
                'dtype': stacked_values.dtype.name,
                'tile_shape': [int(value) for value in tile_shape],
                'compression': effective_compression,
                'compression_level': effective_level,
                'shuffle': effective_shuffle,
                'compression_seconds': None,
                'submission_seconds': submission_seconds,
                'write_seconds': write_seconds,
                'dense_submissions': 1,
                'fragment_count': fragment_count,
                'output_file_count': output_file_count,
                'output_bytes': output_bytes,
            }
        )
    logger.info('Finished writing array %s', uri)
    return uri


def create_empty_scalar_matrix_array(
    base_uri,
    dataset_path,
    n_files,
    n_elements,
    storage_dtype='float32',
    compression='zstd',
    compression_level=5,
    shuffle=True,
    tile_voxels=0,
    target_tile_mb=2.0,
    sources_list: Sequence[str] | None = None,
):
    """Create an empty scalar matrix array in a TileDB directory.

    Parameters
    ----------
    base_uri : :obj:`str`
        Base URI.
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
    tile_voxels : :obj:`int`
        Tile voxels.
    target_tile_mb : :obj:`float`
        Target tile size in MB.
    sources_list : :obj:`list`
        Sources list.

    Returns
    -------
    :obj:`str`
        URI of the created array.
    """
    storage_np_dtype = resolve_dtype(storage_dtype)
    tile_shape = compute_tile_shape_full_subjects(
        n_files, n_elements, tile_voxels, target_tile_mb, storage_np_dtype
    )

    uri = os.path.join(base_uri, dataset_path)
    _ensure_parent_group(uri)

    dim_subjects = tiledb.Dim(
        name='subjects', domain=(0, n_files - 1), tile=tile_shape[0], dtype=_DIM_DTYPE
    )
    dim_items = tiledb.Dim(
        name='items', domain=(0, n_elements - 1), tile=tile_shape[1], dtype=_DIM_DTYPE
    )
    dom = tiledb.Domain(dim_subjects, dim_items)
    attr_filters = _build_filter_list(compression, compression_level, shuffle)
    attr_values = tiledb.Attr(name='values', dtype=storage_np_dtype, filters=attr_filters)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    logger.info(
        'Creating empty TileDB array %s with shape (%d, %d), dtype=%s, tiles=%s',
        uri,
        n_files,
        n_elements,
        storage_np_dtype,
        tile_shape,
    )
    if tiledb.object_type(uri):
        logger.warning('Removing existing array %s', uri)
        tiledb.remove(uri)
    tiledb.Array.create(uri, schema)

    if sources_list is not None:
        try:
            with tiledb.open(uri, 'w') as A:
                A.meta['column_names'] = json.dumps(
                    storage_utils.normalize_column_names(sources_list)
                )
        except (TypeError, ValueError, tiledb.TileDBError):
            logger.warning('Failed to write column_names metadata for %s', uri)
    return uri


# Shared with the other backend; kept under the private name for callers and tests.
_coerce_full_matrix = storage_utils.coerce_full_matrix


def _local_output_stats(uri: str) -> tuple[int | None, int | None]:
    """Return local file count and bytes without treating remote URIs as paths."""
    if '://' in uri or not os.path.exists(uri):
        return None, None
    file_count = 0
    byte_count = 0
    for root, _directories, files in os.walk(uri):
        for filename in files:
            file_count += 1
            try:
                byte_count += os.path.getsize(os.path.join(root, filename))
            except OSError:
                logger.debug('Could not stat TileDB output file %s', filename, exc_info=True)
    return file_count, byte_count


def write_full_matrix(uri: str, values: np.ndarray | Sequence[np.ndarray]) -> dict[str, object]:
    """Write an existing dense TileDB array in exactly one submission."""
    with tiledb.open(uri, 'r') as array_info:
        schema = array_info.schema
        if schema.sparse:
            raise ValueError(f"TileDB array '{uri}' must be dense")
        if schema.domain.ndim != 2:
            raise ValueError(
                f"TileDB array '{uri}' must have two dimensions, got {schema.domain.ndim}"
            )
        if schema.nattr != 1 or not schema.has_attr('values'):
            raise ValueError(
                f"TileDB array '{uri}' must have exactly one attribute named 'values'"
            )

        domain = schema.domain
        shape = tuple(
            int(domain.dim(index).domain[1] - domain.dim(index).domain[0] + 1)
            for index in range(2)
        )
        attr_dtype = schema.attr('values').dtype
        tile_shape = tuple(int(domain.dim(index).tile) for index in range(2))

    matrix = _coerce_full_matrix(values, dtype=attr_dtype, shape=shape)
    write_started = time.perf_counter()
    with tiledb.open(uri, 'w') as array:
        submission_started = time.perf_counter()
        array[:] = {'values': matrix}
        submission_seconds = time.perf_counter() - submission_started
    write_seconds = time.perf_counter() - write_started
    output_file_count, output_bytes = _local_output_stats(uri)
    return {
        'matrix_shape': [int(value) for value in matrix.shape],
        'dtype': matrix.dtype.name,
        'tile_shape': [int(value) for value in tile_shape],
        'compression_seconds': None,
        'submission_seconds': submission_seconds,
        'write_seconds': write_seconds,
        'dense_submissions': 1,
        'fragment_count': len(tiledb.array_fragments(uri)),
        'output_file_count': output_file_count,
        'output_bytes': output_bytes,
    }


def write_rows_in_column_stripes(uri: str, rows: Sequence[np.ndarray]):
    """Compatibility wrapper that now writes rows in one dense submission.

    Parameters
    ----------
    uri : str
        Target array URI with shape (n_files, n_elements).
    rows : Sequence[np.ndarray]
        List/sequence of 1D arrays, one per subject, length == n_elements.
        Each will be cast into one C-contiguous staging matrix if needed.

    Returns
    -------
    dict
        JSON-serializable write metrics from :func:`write_full_matrix`.
    """
    return write_full_matrix(uri, rows)


def write_parcel_names(base_uri: str, array_path: str, names: Sequence[str]):
    """Store parcel names as a 1D dense TileDB string array.

    Parameters
    ----------
    base_uri : str
        Root directory of the TileDB store.
    array_path : str
        Path relative to *base_uri* where the array will be created
        (e.g. ``'parcels/parcel_id'``).
    names : sequence of str
        Parcel name strings to store.
    """
    if len(names) == 0:
        raise ValueError(f"Cannot write parcel names to '{array_path}': names must not be empty.")

    uri = os.path.join(base_uri, array_path)
    _ensure_parent_group(uri)

    n = len(names)
    dim_idx = tiledb.Dim(
        name='idx', domain=(0, max(n - 1, 0)), tile=max(1, min(n, 1024)), dtype=_DIM_DTYPE
    )
    dom = tiledb.Domain(dim_idx)
    # np.unicode_ was removed in NumPy 2.0; np.str_ is the compatible string scalar.
    attr_values = tiledb.Attr(name='values', dtype=np.str_)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    if tiledb.object_type(uri):
        logger.warning('Removing existing array %s', uri)
        tiledb.remove(uri)
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, 'w') as A:
        A[:] = {'values': np.array(names, dtype=object)}


def write_column_names(base_uri: str, scalar: str, sources: Sequence[str]):
    """Store column names as a 1D dense TileDB array for the given scalar.

    Parameters
    ----------
    base_uri : :obj:`str`
        Base URI.
    scalar : :obj:`str`
        Scalar name.
    sources : :obj:`list`
        Sources list.
    """
    scalar = storage_utils.validate_scalar_name(scalar)
    sources = storage_utils.normalize_column_names(sources)
    uri = os.path.join(base_uri, 'scalars', scalar, 'column_names')
    _ensure_parent_group(uri)

    n = len(sources)
    dim_idx = tiledb.Dim(
        name='idx', domain=(0, max(n - 1, 0)), tile=max(1, min(n, 1024)), dtype=_DIM_DTYPE
    )
    dom = tiledb.Domain(dim_idx)
    attr_values = tiledb.Attr(name='values', dtype=np.str_)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    if tiledb.object_type(uri):
        logger.warning('Removing existing array %s', uri)
        tiledb.remove(uri)
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, 'w') as A:
        A[:] = {'values': np.array(sources, dtype=object)}

    # Also write metadata on the parent group for quick discovery (optional)
    group_uri = os.path.join(base_uri, 'scalars', scalar)
    if tiledb.object_type(group_uri):
        try:
            with tiledb.Group(group_uri, 'w') as G:
                G.meta['column_names'] = json.dumps(sources)
        except (TypeError, ValueError, tiledb.TileDBError):
            logger.warning('Failed to write column_names metadata for group %s', group_uri)
