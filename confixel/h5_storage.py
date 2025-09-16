import numpy as np


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
    subjects_per_chunk = num_subjects
    if int(item_chunk) > 0:
        items_per_chunk = min(int(item_chunk), num_items)
    else:
        bytes_per_value = np.dtype(storage_np_dtype).itemsize
        target_bytes = float(target_chunk_mb) * 1024.0 * 1024.0
        items_per_chunk = max(1, int(target_bytes / (bytes_per_value * subjects_per_chunk)))
        items_per_chunk = min(items_per_chunk, num_items)
    return (subjects_per_chunk, items_per_chunk)


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

    dset = h5file.create_dataset(
        dataset_path,
        shape=(num_subjects, num_items),
        dtype=storage_np_dtype,
        chunks=chunk_shape,
        compression=comp,
        compression_opts=comp_opts if comp == 'gzip' else None,
        shuffle=use_shuffle)
    dset[...] = stacked_values
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


