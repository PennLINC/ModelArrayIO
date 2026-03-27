"""Utility functions for fixel-wise data."""

import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_mrconvert():
    return shutil.which('mrconvert')


def _require_mrconvert() -> str:
    mrconvert = find_mrconvert()
    if mrconvert is None:
        raise FileNotFoundError('The mrconvert executable could not be found on $PATH.')
    return mrconvert


def _run_mrconvert(source_file: Path, output_file: Path) -> None:
    try:
        subprocess.run(
            [_require_mrconvert(), str(source_file), str(output_file)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or 'mrconvert failed.'
        raise RuntimeError(
            f'mrconvert failed while converting {source_file} to {output_file}: {message}'
        ) from exc


def nifti2_to_mif(nifti2_image, mif_file):
    """Convert a .nii file to a .mif file.

    Parameters
    ----------
    nifti2_image : :obj:`nibabel.Nifti2Image`
        Nifti2 image
    mif_file : :obj:`str`
        Path to a .mif file
    """
    output_path = Path(mif_file)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_nii = Path(temp_dir) / 'mrconvert_input.nii'
        nifti2_image.to_filename(temp_nii)
        _run_mrconvert(temp_nii, output_path)

    if not output_path.exists():
        raise RuntimeError(f'mrconvert did not create expected output file: {output_path}')


def mif_to_nifti2(mif_file):
    """Convert a .mif file to a .nii file.

    Parameters
    ----------
    mif_file : :obj:`str`
        Path to a .mif file

    Returns
    -------
    nifti2_img : :obj:`nibabel.Nifti2Image`
        Nifti2 image
    data : :obj:`numpy.ndarray`
        Data from the nifti2 image
    """
    input_path = Path(mif_file)
    if input_path.suffix == '.nii':
        nifti2_img = nb.load(input_path)
        data = nifti2_img.get_fdata(dtype=np.float32).squeeze()
        return nifti2_img, data

    with tempfile.TemporaryDirectory() as temp_dir:
        nii_path = Path(temp_dir) / 'mif.nii'
        _run_mrconvert(input_path, nii_path)
        if not nii_path.exists():
            raise RuntimeError(f'mrconvert did not create expected output file: {nii_path}')

        loaded_img = nb.load(nii_path)
        in_memory_data = np.asanyarray(loaded_img.dataobj)
        nifti2_img = nb.Nifti2Image(in_memory_data, loaded_img.affine, header=loaded_img.header)
        data = loaded_img.get_fdata(dtype=np.float32).squeeze()

    return nifti2_img, data


def load_cohort_mif(cohort_long, s3_workers):
    """Load all MIF scalar rows from the cohort, optionally in parallel.

    When s3_workers > 1, a ThreadPoolExecutor is used to run mrconvert
    calls concurrently (subprocess calls release the GIL). Results arrive
    via as_completed and are indexed by (scalar_name, subj_idx) so the
    final ordered lists are reconstructed correctly regardless of completion
    order.

    Parameters
    ----------
    cohort_long : :obj:`pandas.DataFrame`
        Long-format cohort dataframe with columns 'scalar_name' and 'source_file'.
    s3_workers : :obj:`int`
        Number of parallel workers for loading.

    Returns
    -------
    scalars : dict[str, list[np.ndarray]]
        Per-scalar ordered list of 1-D subject arrays, ready for stripe-write.
    sources_lists : dict[str, list[str]]
        Per-scalar ordered list of source file paths (for HDF5 metadata).
    """
    scalar_subj_counter = defaultdict(int)
    jobs = []
    sources_lists = defaultdict(list)

    for row in cohort_long.itertuples(index=False):
        sn = row.scalar_name
        subj_idx = scalar_subj_counter[sn]
        scalar_subj_counter[sn] += 1
        src = row.source_file
        jobs.append((sn, subj_idx, src))
        sources_lists[sn].append(src)

    def _worker(job):
        sn, subj_idx, src = job
        _img, data = mif_to_nifti2(src)
        return sn, subj_idx, data

    if s3_workers > 1:
        results = defaultdict(dict)
        with ThreadPoolExecutor(max_workers=s3_workers) as pool:
            futures = {pool.submit(_worker, job): job for job in jobs}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Loading MIF data',
            ):
                sn, subj_idx, data = future.result()
                results[sn][subj_idx] = data
        scalars = {
            sn: [results[sn][i] for i in range(cnt)] for sn, cnt in scalar_subj_counter.items()
        }
    else:
        scalars = defaultdict(list)
        for job in tqdm(jobs, desc='Loading MIF data'):
            sn, subj_idx, data = _worker(job)
            scalars[sn].append(data)

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
    _index_img, index_data = mif_to_nifti2(index_file)
    # number of fixels in each voxel; by index.mif definition
    count_vol = index_data[..., 0].astype(np.uint32)
    # index of the first fixel in this voxel, in the list of all fixels
    # (in directions.mif, FD.mif, etc)
    id_vol = index_data[..., 1]
    max_id = id_vol.max()
    # = the maximum id of fixels + 1 = # of fixels in entire image
    max_fixel_id = max_id + int(count_vol[id_vol == max_id])
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

    _directions_img, directions_data = mif_to_nifti2(directions_file)
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
