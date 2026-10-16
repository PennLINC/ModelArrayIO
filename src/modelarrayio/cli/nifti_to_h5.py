"""Convert NIfTI data to an HDF5 file."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.runtime import (
    TimingRecorder,
    publish_scalar_outputs,
    resolve_loader_workers,
    validate_timing_jsonl_path,
)
from modelarrayio.storage import utils as storage_utils
from modelarrayio.utils.nifti import load_cohort_voxels
from modelarrayio.utils.s3_utils import (
    configure_s3_client_pool,
    load_nibabel,
    release_image_buffers,
)

logger = logging.getLogger(__name__)


def nifti_to_h5(
    group_mask_file,
    cohort_long,
    backend='hdf5',
    output=Path('voxelarray.h5'),
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=None,
    s3_workers=None,
    split_outputs=False,
    timing_jsonl=None,
):
    """Load all volume data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    group_mask_file : :obj:`str`
        Path to a NIfTI-1 binary group mask file.
    cohort_long : :obj:`pandas.DataFrame`
        Normalised long-format cohort dataframe (from
        :func:`~modelarrayio.utils.misc.load_and_normalize_cohort`).
    backend : :obj:`str`
        Storage backend (``'hdf5'`` or ``'tiledb'``).
    output : :obj:`pathlib.Path`
        Output path. For the hdf5 backend, path to an .h5 file;
        for the tiledb backend, path to a .tdb directory.
    storage_dtype : :obj:`str`
        Floating type to store values. Options: ``'float32'`` (default), ``'float64'``.
    compression : :obj:`str`
        Compression filter. ``gzip`` works for both backends;
        ``lzf`` is HDF5-only; ``zstd`` is TileDB-only.
    compression_level : :obj:`int`
        Compression level (codec-dependent). Default 4.
    shuffle : :obj:`bool`
        Enable shuffle filter. Default True.
    chunk_voxels : :obj:`int`
        Chunk/tile size along the voxel axis. If 0, auto-compute. Default 0.
    target_chunk_mb : :obj:`float`
        Target chunk/tile size in MiB when auto-computing. Default 2.0.
    workers : :obj:`int`
        Maximum number of input sources loaded concurrently, for every backend.
    s3_workers : :obj:`int`
        Deprecated alias for ``workers``.
    split_outputs : :obj:`bool`
        If True, write one output file per scalar. Default False.
    timing_jsonl : path-like, optional
        Append structured per-scalar and total runtime records to this file.
    """
    worker_count = resolve_loader_workers(workers, s3_workers)
    configure_s3_client_pool(worker_count)
    compression, compression_level = storage_utils.resolve_backend_compression(
        backend, compression, compression_level
    )
    planned_scalar_names = list(dict.fromkeys(str(value) for value in cohort_long['scalar_name']))
    planned_outputs = cli_utils.conversion_output_paths(
        output,
        planned_scalar_names,
        split_outputs=split_outputs,
    )
    validate_timing_jsonl_path(
        timing_jsonl,
        planned_outputs,
        directory_outputs=backend == 'tiledb',
    )
    recorder = TimingRecorder(
        timing_jsonl,
        modality='nifti',
        backend=backend,
        workers=worker_count,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
    )
    group_mask_img = load_nibabel(group_mask_file)
    try:
        group_mask_matrix = np.asanyarray(group_mask_img.dataobj) > 0
    finally:
        release_image_buffers(group_mask_img)
    voxel_coords = np.column_stack(np.nonzero(group_mask_matrix))

    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(voxel_coords.shape[0]),
            'i': voxel_coords[:, 0],
            'j': voxel_coords[:, 1],
            'k': voxel_coords[:, 2],
        }
    )

    logger.info('Extracting NIfTI data...')
    scalars, sources_lists = load_cohort_voxels(
        cohort_long,
        group_mask_matrix,
        workers=worker_count,
        storage_dtype=storage_dtype,
        group_mask_path=group_mask_file,
    )
    if not sources_lists:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    def _hdf5_metadata(h5_file):
        cli_utils.write_table_dataset(h5_file, 'voxels', voxel_table)

    publish_scalar_outputs(
        recorder,
        backend=backend,
        output=output,
        split_outputs=split_outputs,
        scalar_names=list(sources_lists),
        scalars=scalars,
        sources_by_scalar=sources_lists,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
        expected_elements=len(voxel_table),
        hdf5_metadata=_hdf5_metadata,
        hdf5_required_paths=('voxels',),
    )
    return 0
