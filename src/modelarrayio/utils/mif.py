"""Utility functions for MIF data."""

from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.filebasedimages import FileHolder

from modelarrayio.utils.loading import (
    LazyScalarMatrices,
    resolve_loader_workers,
    stage_source_rows,
)
from modelarrayio.utils.mif_image import MifHeader, MifImage
from modelarrayio.utils.s3_utils import (
    _check_local_source,
    is_s3_path,
    read_s3_bytes,
    release_image_buffers,
)

__all__ = ['MifHeader', 'MifImage', 'gather_fixels', 'load_cohort_mif', 'mif_to_image']


def _load_mif_image(mif_file):
    """Load a local, materialized-annex, or S3 MIF image."""
    if not is_s3_path(mif_file):
        return MifImage.from_filename(str(_check_local_source(Path(mif_file))))

    data = read_s3_bytes(mif_file, decompress_gzip=True)
    file_map = {'image': FileHolder(fileobj=BytesIO(data))}
    return MifImage.from_file_map(file_map)


def mif_to_image(mif_file, dtype=np.float32):
    """Load a `.mif` file into a :class:`MifImage`.

    Parameters
    ----------
    mif_file : :obj:`str`
        Path to a .mif file

    Returns
    -------
    mif_img : :obj:`MifImage`
        Loaded MIF image
    data : :obj:`numpy.ndarray`
        Data from the MIF image
    """
    mif_img = _load_mif_image(mif_file)
    release_after_decode = is_s3_path(mif_file)
    try:
        if hasattr(mif_img, 'dataobj'):
            data = np.asanyarray(mif_img.dataobj).squeeze()
            if dtype is not None and data.dtype != np.dtype(dtype):
                data = data.astype(dtype, copy=False)
        else:
            # Compatibility with nibabel-like test doubles and older readers.
            data = mif_img.get_fdata(dtype=dtype).squeeze()
    finally:
        if release_after_decode:
            release_image_buffers(mif_img)
    return mif_img, data


def load_cohort_mif(
    cohort_long,
    s3_workers=None,
    *,
    workers=None,
    storage_dtype='float32',
):
    """Return lazy per-scalar MIF staging matrices and ordered sources.

    Only the matrix requested by the writer is resident. Its rows are loaded
    with the shared bounded thread scheduler and cast once as they enter the
    requested staging dtype.

    Parameters
    ----------
    cohort_long : :obj:`pandas.DataFrame`
        Long-format cohort dataframe with columns 'scalar_name' and 'source_file'.
    s3_workers : :obj:`int`
        Deprecated positional alias for the loader worker count.
    workers : :obj:`int`
        Maximum number of concurrent source loads.
    storage_dtype : {'float32', 'float64'}
        Dtype of the staging matrix.

    Returns
    -------
    scalars : modelarrayio.utils.loading.LazyScalarMatrices
        Lazy mapping that stages one ordered, typed matrix per scalar.
    sources_lists : dict[str, list[str]]
        Per-scalar ordered list of source file paths (for HDF5 metadata).
    """
    worker_count = resolve_loader_workers(workers, s3_workers)

    sources_lists: OrderedDict[str, list[str]] = OrderedDict()
    for row in cohort_long.itertuples(index=False):
        sources_lists.setdefault(str(row.scalar_name), []).append(str(row.source_file))

    def _load_native(source):
        _image, data = mif_to_image(source, dtype=None)
        return np.asarray(data).squeeze()

    def _stage(scalar_name):
        return stage_source_rows(
            sources_lists[scalar_name],
            _load_native,
            scalar_name=scalar_name,
            storage_dtype=storage_dtype,
            workers=worker_count,
        )

    scalars = LazyScalarMatrices(list(sources_lists), _stage, sources_by_scalar=sources_lists)
    return scalars, sources_lists


def gather_fixels(index_file, directions_file):
    """Load the index and directions files to get lookup tables.

    Parameters
    ----------
    index_file : :obj:`str`
        Path to a Nifti2 index file
    directions_file : :obj:`str`
        Path to a Nifti2 directions file

    Returns
    -------
    fixel_table : :obj:`pandas.DataFrame`
        DataFrame with fixel_id, voxel_id, x, y, z
    voxel_table : :obj:`pandas.DataFrame`
        DataFrame with voxel_id, i, j, k
    """
    _index_img, index_data = mif_to_image(index_file)
    # number of fixels in each voxel; by index.mif definition
    count_vol = index_data[..., 0].astype(np.uint32)
    # index of the first fixel in this voxel, in the list of all fixels
    # (in directions.mif, FD.mif, etc)
    id_vol = index_data[..., 1]
    max_id = id_vol.max()
    # = the maximum id of fixels + 1 = # of fixels in entire image
    terminal_counts = count_vol[id_vol == max_id]
    if terminal_counts.size == 0:
        raise ValueError('Could not determine the final fixel count from the index image.')
    max_fixel_id = int(max_id) + int(terminal_counts.max())
    voxel_mask = count_vol > 0  # voxels that contains fixel(s), =1
    masked_ids = id_vol[voxel_mask]  # 1D array, len = # of voxels with fixel(s), value see id_vol
    masked_counts = count_vol[voxel_mask]  # dim as masked_ids; value see count_vol
    # indices that would sort array masked_ids value (i.e. first fixel's id in this voxel) from
    # lowest to highest; so it's sorting voxels by their first fixel id
    id_sort = np.argsort(masked_ids)
    sorted_counts = masked_counts[id_sort]
    # dim: [# of voxels with fixel(s)] x 3, each row is the subscript i.e. (i,j,k) in 3D
    # image of a voxel with fixel
    voxel_coords = np.column_stack(np.nonzero(count_vol))

    fixel_id = 0
    fixel_ids = np.arange(max_fixel_id, dtype=np.int32)
    fixel_voxel_ids = np.zeros_like(fixel_ids)
    for voxel_id, fixel_count in enumerate(sorted_counts):
        for _ in range(fixel_count):
            # fixel_voxel_ids: 1D, len = # of fixels; each value is the voxel_id of the voxel
            # where this fixel locates
            fixel_voxel_ids[fixel_id] = voxel_id
            fixel_id += 1
    sorted_coords = voxel_coords[id_sort]

    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(voxel_coords.shape[0]),
            'i': sorted_coords[:, 0],
            'j': sorted_coords[:, 1],
            'k': sorted_coords[:, 2],
        }
    )

    _directions_img, directions_data = mif_to_image(directions_file)
    fixel_table = pd.DataFrame(
        {
            'fixel_id': fixel_ids,
            'voxel_id': fixel_voxel_ids,
            'x': directions_data[:, 0],
            'y': directions_data[:, 1],
            'z': directions_data[:, 2],
        }
    )

    return fixel_table, voxel_table
