"""HDF5 storage utilities."""

from __future__ import annotations

import logging

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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

    if stacked_values.dtype != storage_np_dtype:
        stacked_values = stacked_values.astype(storage_np_dtype)

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
    dset[...] = stacked_values
    logger.info('Finished writing dataset %s', dataset_path)
    if sources_list is not None:
        dset.attrs['column_names'] = list(sources_list)
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
    sources_list=None | pd.Series | list,
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
        # dataset_path is e.g. 'scalars/FA/values'; extract the scalar name segment
        scalar_name = dataset_path.split('/')[1] if dataset_path.count('/') >= 2 else dataset_path
        write_column_names(h5file, scalar_name, sources_list)
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
    values = np.array(storage_utils.normalize_column_names(sources), dtype=object)
    grp = h5_file.require_group(f'scalars/{scalar}')

    # Variable-length UTF-8 string dtype
    vlen_str = h5py.string_dtype(encoding='utf-8')

    # Create 1-D dataset of strings
    grp.create_dataset(
        'column_names',
        data=values,
        dtype=vlen_str,
        shape=(len(values),),
    )


def write_rows_in_column_stripes(dset, rows):
    """Fill a 2D HDF5 dataset by buffering column-aligned stripes to minimize
    chunk recompression, using about one chunk's worth of memory.

    Parameters
    ----------
    dset : h5py.Dataset
        Target dataset with shape (n_files, n_elements) and chunking set.
    rows : Sequence[np.ndarray]
        List/sequence of 1D arrays, one per subject, length == n_elements.
        Each will be cast on write to dset.dtype if needed.
    """
    n_files, n_elements = dset.shape
    if len(rows) != n_files:
        raise ValueError('rows length does not match dataset subjects dimension')
    stripe_width = dset.chunks[1] if dset.chunks is not None else max(1, n_elements // 8)
    logger.info(
        'Stripe-writing dataset %s with stripe width=%d (chunks=%s)',
        dset.name,
        stripe_width,
        str(dset.chunks),
    )

    buf = np.empty((n_files, stripe_width), dtype=dset.dtype)
    with logging_redirect_tqdm():
        for start in tqdm(
            range(0, n_elements, stripe_width),
            bar_format=(
                '{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ),
            ascii=True,
            mininterval=max(1, (n_elements / stripe_width) // 200),
        ):
            end = min(start + stripe_width, n_elements)
            width = end - start
            if width != stripe_width:
                # resize buffer view on last partial stripe
                buf_view = buf[:, :width]
            else:
                buf_view = buf
            for i, row in enumerate(rows):
                # slice is contiguous; cast on assignment if needed
                buf_view[i, :] = row[start:end]
            logger.debug('Writing stripe [%d:%d] to %s', start, end, dset.name)
            dset[:, start:end] = buf_view
    logger.info('Finished stripe-writing dataset %s', dset.name)
