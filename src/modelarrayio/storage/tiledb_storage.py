"""TileDB storage utilities."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence

import numpy as np
import tiledb

from modelarrayio.storage import utils as storage_utils

logger = logging.getLogger(__name__)


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


def _build_filter_list(compression: str | None, compression_level: int | None, shuffle: bool):
    filters = []
    if shuffle:
        # ByteShuffle works well for float data; BitShuffle is also available
        filters.append(tiledb.ByteShuffleFilter())
    if compression is None or str(compression).lower() == 'none':
        pass
    else:
        comp = str(compression).lower()
        level = None
        try:
            level = int(compression_level) if compression_level is not None else None
        except (TypeError, ValueError):
            level = None
        if comp == 'zstd':
            filters.append(tiledb.ZstdFilter(level=level if level is not None else 5))
        elif comp == 'gzip':
            filters.append(tiledb.GzipFilter(level=level if level is not None else 4))
        else:
            # Fallback: no compression if an unknown codec is provided
            logger.warning("Unknown compression '%s' for TileDB; disabling compression.", comp)
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
    if parent and not tiledb.object_type(parent):
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
    if stacked_values.dtype != storage_np_dtype:
        stacked_values = stacked_values.astype(storage_np_dtype)

    n_files, n_elements = stacked_values.shape
    tile_shape = compute_tile_shape_full_subjects(
        n_files, n_elements, tile_voxels, target_tile_mb, storage_np_dtype
    )

    uri = os.path.join(base_uri, dataset_path)
    _ensure_parent_group(uri)

    # Domain and schema
    dim_subjects = tiledb.Dim(
        name='subjects', domain=(0, n_files - 1), tile=tile_shape[0], dtype=np.int64
    )
    dim_items = tiledb.Dim(
        name='items', domain=(0, n_elements - 1), tile=tile_shape[1], dtype=np.int64
    )
    dom = tiledb.Domain(dim_subjects, dim_items)
    attr_filters = _build_filter_list(compression, compression_level, shuffle)
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
        tiledb.remove(uri)
    tiledb.Array.create(uri, schema)

    logger.info('Writing full array %s to TileDB (this may take a while)...', uri)
    with tiledb.open(uri, 'w') as A:
        A[:] = {'values': stacked_values}
        if sources_list is not None:
            try:
                A.meta['column_names'] = json.dumps(
                    storage_utils.normalize_column_names(sources_list)
                )
            except (TypeError, ValueError, tiledb.TileDBError):
                # Fallback without metadata if serialization fails
                logger.warning('Failed to write column_names metadata for %s', uri)
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
        name='subjects', domain=(0, n_files - 1), tile=tile_shape[0], dtype=np.int64
    )
    dim_items = tiledb.Dim(
        name='items', domain=(0, n_elements - 1), tile=tile_shape[1], dtype=np.int64
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


def write_rows_in_column_stripes(uri: str, rows: Sequence[np.ndarray]):
    """Fill a 2D TileDB dense array by buffering column-aligned stripes to minimize
    tile writes, using about one tile's worth of memory.

    Parameters
    ----------
    uri : str
        Target array URI with shape (n_files, n_elements).
    rows : Sequence[np.ndarray]
        List/sequence of 1D arrays, one per subject, length == n_elements.
        Each will be cast on write to array attr dtype if needed.
    """
    with tiledb.open(uri, 'r') as Ainfo:
        dom = Ainfo.schema.domain
        n_files = dom.dim(0).domain[1] - dom.dim(0).domain[0] + 1
        n_elements = dom.dim(1).domain[1] - dom.dim(1).domain[0] + 1
        attr_dtype = Ainfo.schema.attr(0).dtype

    if len(rows) != n_files:
        raise ValueError('rows length does not match array subjects dimension')

    # Try to align stripe width to the items tile for best throughput
    with tiledb.open(uri, 'r') as Ainfo2:
        items_tile = Ainfo2.schema.domain.dim(1).tile
    stripe_width = items_tile if items_tile is not None else max(1, n_elements // 8)

    buf = np.empty((n_files, stripe_width), dtype=attr_dtype)
    for start in range(0, n_elements, stripe_width):
        end = min(start + stripe_width, n_elements)
        width = end - start
        if width != stripe_width:
            buf_view = buf[:, :width]
        else:
            buf_view = buf
        for i, row in enumerate(rows):
            buf_view[i, :] = row[start:end]
        with tiledb.open(uri, 'w') as A:
            A[:, start:end] = {'values': buf_view}


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
        name='idx', domain=(0, max(n - 1, 0)), tile=max(1, min(n, 1024)), dtype=np.int64
    )
    dom = tiledb.Domain(dim_idx)
    attr_values = tiledb.Attr(name='values', dtype=np.unicode_)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    if tiledb.object_type(uri):
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
    sources = storage_utils.normalize_column_names(sources)
    uri = os.path.join(base_uri, 'scalars', scalar, 'column_names')
    _ensure_parent_group(uri)

    n = len(sources)
    dim_idx = tiledb.Dim(
        name='idx', domain=(0, max(n - 1, 0)), tile=max(1, min(n, 1024)), dtype=np.int64
    )
    dom = tiledb.Domain(dim_idx)
    attr_values = tiledb.Attr(name='values', dtype=np.str_)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    if tiledb.object_type(uri):
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
