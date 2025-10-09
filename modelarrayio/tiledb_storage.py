import os
import json
import logging
from typing import Sequence

import numpy as np
import tiledb

logger = logging.getLogger(__name__)


def resolve_dtype(storage_dtype):
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
    }
    return dtype_map.get(str(storage_dtype).lower(), np.float32)


def _build_filter_list(compression: str | None, compression_level: int | None, shuffle: bool):
    filters = []
    if shuffle:
        # ByteShuffle works well for float data; BitShuffle is also available
        filters.append(tiledb.ByteShuffleFilter())
    if compression is None or str(compression).lower() == "none":
        pass
    else:
        comp = str(compression).lower()
        level = None
        try:
            level = int(compression_level) if compression_level is not None else None
        except Exception:
            level = None
        if comp == "zstd":
            filters.append(tiledb.ZstdFilter(level=level if level is not None else 5))
        elif comp == "gzip":
            filters.append(tiledb.GzipFilter(level=level if level is not None else 4))
        else:
            # Fallback: no compression if an unknown codec is provided
            logger.warning("Unknown compression '%s' for TileDB; disabling compression.", comp)
    return tiledb.FilterList(filters)


def compute_tile_shape_full_subjects(
    num_subjects, num_items, item_tile, target_tile_mb, storage_np_dtype
):
    num_subjects = int(num_subjects)
    num_items = int(num_items)
    if num_subjects <= 0 or num_items <= 0:
        raise ValueError(
            f"Cannot compute tile shape with zero-length dimension: num_subjects={num_subjects}, num_items={num_items}"
        )

    subjects_per_tile = num_subjects
    if int(item_tile) > 0:
        items_per_tile = min(int(item_tile), num_items)
    else:
        bytes_per_value = np.dtype(storage_np_dtype).itemsize
        target_bytes = float(target_tile_mb) * 1024.0 * 1024.0
        items_per_tile = max(1, int(target_bytes / (bytes_per_value * subjects_per_tile)))
        items_per_tile = min(items_per_tile, num_items)
    tile = (subjects_per_tile, items_per_tile)
    logger.debug(
        "Computed tile shape: %s (subjects=%d, items=%d, item_tile=%s, target_tile_mb=%.2f)",
        tile,
        num_subjects,
        num_items,
        str(item_tile),
        float(target_tile_mb),
    )
    return tile


def _ensure_parent_group(uri: str):
    parent = os.path.dirname(uri.rstrip("/"))
    if parent and not tiledb.object_type(parent):
        tiledb.group_create(parent)


def create_scalar_matrix_array(
    base_uri,
    dataset_path,
    stacked_values,
    sources_list,
    storage_dtype="float32",
    compression="zstd",
    compression_level=5,
    shuffle=True,
    tile_voxels=0,
    target_tile_mb=2.0,
):
    storage_np_dtype = resolve_dtype(storage_dtype)
    if stacked_values.dtype != storage_np_dtype:
        stacked_values = stacked_values.astype(storage_np_dtype)

    num_subjects, num_items = stacked_values.shape
    tile_shape = compute_tile_shape_full_subjects(
        num_subjects, num_items, tile_voxels, target_tile_mb, storage_np_dtype
    )

    uri = os.path.join(base_uri, dataset_path)
    _ensure_parent_group(uri)

    # Domain and schema
    dim_subjects = tiledb.Dim(name="subjects", domain=(0, num_subjects - 1), tile=tile_shape[0], dtype=np.int64)
    dim_items = tiledb.Dim(name="items", domain=(0, num_items - 1), tile=tile_shape[1], dtype=np.int64)
    dom = tiledb.Domain(dim_subjects, dim_items)
    attr_filters = _build_filter_list(compression, compression_level, shuffle)
    attr_values = tiledb.Attr(name="values", dtype=storage_np_dtype, filters=attr_filters)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    logger.info(
        "Creating TileDB array %s with shape (%d, %d), dtype=%s, tiles=%s",
        uri,
        num_subjects,
        num_items,
        storage_np_dtype,
        tile_shape,
    )
    tiledb.Array.create(uri, schema)

    logger.info("Writing full array %s to TileDB (this may take a while)...", uri)
    with tiledb.open(uri, "w") as A:
        A[:] = {"values": stacked_values}
        if sources_list is not None:
            try:
                A.meta["column_names"] = json.dumps(list(sources_list))
            except Exception:
                # Fallback without metadata if serialization fails
                logger.warning("Failed to write column_names metadata for %s", uri)
    logger.info("Finished writing array %s", uri)
    return uri


def create_empty_scalar_matrix_array(
    base_uri,
    dataset_path,
    num_subjects,
    num_items,
    storage_dtype="float32",
    compression="zstd",
    compression_level=5,
    shuffle=True,
    tile_voxels=0,
    target_tile_mb=2.0,
    sources_list: Sequence[str] | None = None,
):
    storage_np_dtype = resolve_dtype(storage_dtype)
    tile_shape = compute_tile_shape_full_subjects(
        num_subjects, num_items, tile_voxels, target_tile_mb, storage_np_dtype
    )

    uri = os.path.join(base_uri, dataset_path)
    _ensure_parent_group(uri)

    dim_subjects = tiledb.Dim(name="subjects", domain=(0, num_subjects - 1), tile=tile_shape[0], dtype=np.int64)
    dim_items = tiledb.Dim(name="items", domain=(0, num_items - 1), tile=tile_shape[1], dtype=np.int64)
    dom = tiledb.Domain(dim_subjects, dim_items)
    attr_filters = _build_filter_list(compression, compression_level, shuffle)
    attr_values = tiledb.Attr(name="values", dtype=storage_np_dtype, filters=attr_filters)
    schema = tiledb.ArraySchema(domain=dom, attrs=[attr_values], sparse=False)

    logger.info(
        "Creating empty TileDB array %s with shape (%d, %d), dtype=%s, tiles=%s",
        uri,
        num_subjects,
        num_items,
        storage_np_dtype,
        tile_shape,
    )
    tiledb.Array.create(uri, schema)

    if sources_list is not None:
        try:
            with tiledb.open(uri, "w") as A:
                A.meta["column_names"] = json.dumps(list(map(str, sources_list)))
        except Exception:
            logger.warning("Failed to write column_names metadata for %s", uri)
    return uri


def write_rows_in_column_stripes(uri: str, rows: Sequence[np.ndarray]):
    """
    Fill a 2D TileDB dense array by buffering column-aligned stripes to minimize
    tile writes, using about one tile's worth of memory.

    Parameters
    ----------
    uri : str
        Target array URI with shape (num_subjects, num_elements).
    rows : Sequence[np.ndarray]
        List/sequence of 1D arrays, one per subject, length == num_elements.
        Each will be cast on write to array attr dtype if needed.
    """
    with tiledb.open(uri, "r") as Ainfo:
        dom = Ainfo.schema.domain
        num_subjects = dom.dim(0).domain[1] - dom.dim(0).domain[0] + 1
        num_elements = dom.dim(1).domain[1] - dom.dim(1).domain[0] + 1
        attr_dtype = Ainfo.schema.attr(0).dtype

    if len(rows) != num_subjects:
        raise ValueError("rows length does not match array subjects dimension")

    # Try to align stripe width to the items tile for best throughput
    with tiledb.open(uri, "r") as Ainfo2:
        items_tile = Ainfo2.schema.domain.dim(1).tile
    stripe_width = items_tile if items_tile is not None else max(1, num_elements // 8)

    buf = np.empty((num_subjects, stripe_width), dtype=attr_dtype)
    for start in range(0, num_elements, stripe_width):
        end = min(start + stripe_width, num_elements)
        width = end - start
        if width != stripe_width:
            buf_view = buf[:, :width]
        else:
            buf_view = buf
        for i, row in enumerate(rows):
            buf_view[i, :] = row[start:end]
        with tiledb.open(uri, "w") as A:
            A[:, start:end] = {"values": buf_view}


def write_column_names(base_uri: str, scalar: str, sources: Sequence[str]):
    """
    Store column names as metadata on the TileDB group for the given scalar.
    This mirrors HDF5's practice of storing names alongside the data.
    """
    group_uri = os.path.join(base_uri, "scalars", scalar)
    if not tiledb.object_type(group_uri):
        tiledb.group_create(group_uri)
    with tiledb.Group(group_uri, "w") as G:
        try:
            G.meta["column_names"] = json.dumps(list(map(str, sources)))
        except Exception:
            logger.warning("Failed to write column_names metadata for group %s", group_uri)


