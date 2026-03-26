"""Utility functions for voxel-wise data."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nb
import numpy as np
from tqdm import tqdm

from modelarrayio.utils.s3_utils import load_nibabel


def _load_cohort_voxels(cohort_df, group_mask_matrix, s3_workers):
    """Load all voxel rows from the cohort, optionally in parallel.

    When s3_workers > 1, a ThreadPoolExecutor is used. Threads share memory so
    group_mask_matrix is accessed directly with no copying overhead. Results
    arrive via as_completed and are indexed by (scalar_name, subj_idx) so the
    final ordered lists are reconstructed correctly regardless of completion order.

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

    for _, row in cohort_df.iterrows():
        sn = row['scalar_name']
        subj_idx = scalar_subj_counter[sn]
        scalar_subj_counter[sn] += 1
        src = row['source_file']
        msk = row['source_mask_file']
        jobs.append((sn, subj_idx, src, msk))
        sources_lists[sn].append(src)

    def _worker(job):
        sn, subj_idx, scalar_path, mask_path = job
        scalar_img = load_nibabel(scalar_path)
        mask_img = load_nibabel(mask_path)
        arr = flattened_image(scalar_img, mask_img, group_mask_matrix)
        return sn, subj_idx, arr

    if s3_workers > 1:
        results = defaultdict(dict)
        with ThreadPoolExecutor(max_workers=s3_workers) as pool:
            futures = {pool.submit(_worker, job): job for job in jobs}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Loading voxel data',
            ):
                sn, subj_idx, arr = future.result()
                results[sn][subj_idx] = arr
        scalars = {
            sn: [results[sn][i] for i in range(cnt)] for sn, cnt in scalar_subj_counter.items()
        }
    else:
        scalars = defaultdict(list)
        for job in tqdm(jobs, desc='Loading voxel data'):
            sn, subj_idx, arr = _worker(job)
            scalars[sn].append(arr)

    return scalars, sources_lists


def flattened_image(scalar_image, scalar_mask, group_mask_matrix):
    scalar_mask_img = scalar_mask if hasattr(scalar_mask, 'get_fdata') else nb.load(scalar_mask)
    scalar_mask_matrix = scalar_mask_img.get_fdata() > 0

    scalar_img = scalar_image if hasattr(scalar_image, 'get_fdata') else nb.load(scalar_image)
    scalar_matrix = scalar_img.get_fdata()

    scalar_matrix[np.logical_not(scalar_mask_matrix)] = np.nan
    # .shape = (#voxels,)  # squeeze() is to remove the 2nd dimension which is not necessary
    return scalar_matrix[group_mask_matrix].squeeze()
