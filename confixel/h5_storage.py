import numpy as np
import logging

logger = logging.getLogger(__name__)


def resolve_dtype(storage_dtype):
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
    }
    return dtype_map.get(str(storage_dtype).lower(), np.float32)


def resolve_compression(compression, compression_level, shuffle):
    comp = None if compression is None or str(compression).lower() == 'none' else str(compression).lower()
    use_shuffle = bool(shuffle) if comp is not None else False
    gzip_level = None
    if comp == 'gzip':
        try:
            gzip_level = int(compression_level)
        except Exception:
            gzip_level = 4
        gzip_level = max(0, min(9, gzip_level))
    return comp, gzip_level, use_shuffle


def compute_chunk_shape_full_subjects(num_subjects, num_items, item_chunk, target_chunk_mb, storage_np_dtype):
    # Fail fast on zero-sized dimensions to avoid invalid chunk shapes and division by zero
    num_subjects = int(num_subjects)
    num_items = int(num_items)
    if num_subjects <= 0 or num_items <= 0:
        raise ValueError(
            f"Cannot compute chunk shape with zero-length dimension: num_subjects={num_subjects}, num_items={num_items}"
        )

    subjects_per_chunk = num_subjects
    if int(item_chunk) > 0:
        items_per_chunk = min(int(item_chunk), num_items)
    else:
        bytes_per_value = np.dtype(storage_np_dtype).itemsize
        target_bytes = float(target_chunk_mb) * 1024.0 * 1024.0
        items_per_chunk = max(1, int(target_bytes / (bytes_per_value * subjects_per_chunk)))
        items_per_chunk = min(items_per_chunk, num_items)
    chunk = (subjects_per_chunk, items_per_chunk)
    logger.debug("Computed chunk shape: %s (subjects=%d, items=%d, item_chunk=%s, target_chunk_mb=%.2f)",
                 chunk, num_subjects, num_items, str(item_chunk), float(target_chunk_mb))
    return chunk


def create_scalar_matrix_dataset(h5file, dataset_path, stacked_values, sources_list,
                                 storage_dtype='float32', compression='gzip', compression_level=4,
                                 shuffle=True, chunk_voxels=0, target_chunk_mb=2.0):
    storage_np_dtype = resolve_dtype(storage_dtype)
    comp, comp_opts, use_shuffle = resolve_compression(compression, compression_level, shuffle)

    if stacked_values.dtype != storage_np_dtype:
        stacked_values = stacked_values.astype(storage_np_dtype)

    num_subjects, num_items = stacked_values.shape
    chunk_shape = compute_chunk_shape_full_subjects(num_subjects, num_items, chunk_voxels,
                                                    target_chunk_mb, storage_np_dtype)
    logger.info("Creating dataset %s with shape (%d, %d), dtype=%s, chunks=%s, compression=%s",
                dataset_path, num_subjects, num_items, storage_np_dtype, chunk_shape, str(comp))
    dset = h5file.create_dataset(
        dataset_path,
        shape=(num_subjects, num_items),
        dtype=storage_np_dtype,
        chunks=chunk_shape,
        compression=comp,
        compression_opts=comp_opts if comp == 'gzip' else None,
        shuffle=use_shuffle)
    logger.info("Writing full dataset %s to HDF5 (this may take a while)...", dataset_path)
    dset[...] = stacked_values
    logger.info("Finished writing dataset %s", dataset_path)
    if sources_list is not None:
        dset.attrs['column_names'] = list(sources_list)
    return dset


def create_empty_scalar_matrix_dataset(h5file, dataset_path, num_subjects, num_items,
                                       storage_dtype='float32', compression='gzip', compression_level=4,
                                       shuffle=True, chunk_voxels=0, target_chunk_mb=2.0, sources_list=None):
    storage_np_dtype = resolve_dtype(storage_dtype)
    comp, comp_opts, use_shuffle = resolve_compression(compression, compression_level, shuffle)

    chunk_shape = compute_chunk_shape_full_subjects(num_subjects, num_items, chunk_voxels,
                                                    target_chunk_mb, storage_np_dtype)
    logger.info("Creating empty dataset %s with shape (%d, %d), dtype=%s, chunks=%s, compression=%s",
                dataset_path, num_subjects, num_items, storage_np_dtype, chunk_shape, str(comp))
    dset = h5file.create_dataset(
        dataset_path,
        shape=(num_subjects, num_items),
        dtype=storage_np_dtype,
        chunks=chunk_shape,
        compression=comp,
        compression_opts=comp_opts if comp == 'gzip' else None,
        shuffle=use_shuffle)
    if sources_list is not None:
        dset.attrs['column_names'] = list(sources_list)
    return dset



def write_rows_in_column_stripes(dset, rows):
    """
    Fill a 2D HDF5 dataset by buffering column-aligned stripes to minimize
    chunk recompression, using about one chunk's worth of memory.

    Parameters
    ----------
    dset : h5py.Dataset
        Target dataset with shape (num_subjects, num_items) and chunking set.
    rows : Sequence[np.ndarray]
        List/sequence of 1D arrays, one per subject, length == num_items.
        Each will be cast on write to dset.dtype if needed.
    """
    num_subjects, num_items = dset.shape
    if len(rows) != num_subjects:
        raise ValueError("rows length does not match dataset subjects dimension")
    stripe_width = dset.chunks[1] if dset.chunks is not None else max(1, num_items // 8)
    logger.info("Stripe-writing dataset %s with stripe width=%d (chunks=%s)",
                dset.name, stripe_width, str(dset.chunks))

    buf = np.empty((num_subjects, stripe_width), dtype=dset.dtype)
    for start in range(0, num_items, stripe_width):
        end = min(start + stripe_width, num_items)
        width = end - start
        if width != stripe_width:
            # resize buffer view on last partial stripe
            buf_view = buf[:, :width]
        else:
            buf_view = buf
        for i, row in enumerate(rows):
            # slice is contiguous; cast on assignment if needed
            buf_view[i, :] = row[start:end]
        logger.debug("Writing stripe [%d:%d] to %s", start, end, dset.name)
        dset[:, start:end] = buf_view
    logger.info("Finished stripe-writing dataset %s", dset.name)

