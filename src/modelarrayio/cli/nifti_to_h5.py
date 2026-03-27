"""Convert NIfTI data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd

from modelarrayio.cli import diagnostics as cli_diagnostics
from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import _is_file, add_diagnostics_args, add_to_modelarray_args
from modelarrayio.utils.voxels import _load_cohort_voxels

logger = logging.getLogger(__name__)


def nifti_to_h5(
    group_mask_file,
    cohort_file,
    backend='hdf5',
    output=Path('voxelarray.h5'),
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=None,
    s3_workers=1,
    no_diagnostics=False,
    diagnostics_dir=None,
    diagnostic_maps=None,
):
    """Load all volume data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    group_mask_file : :obj:`str`
        Path to a NIfTI-1 binary group mask file.
    cohort_file : :obj:`str`
        Path to a CSV with demographic info and paths to data.
    backend : :obj:`str`
        Storage backend (``'hdf5'`` or ``'tiledb'``).
    output : :obj:`str`
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
        Maximum number of parallel TileDB write workers. Default 0 (auto).
        Has no effect when ``backend='hdf5'``.
    s3_workers : :obj:`int`
        Number of parallel workers for S3 downloads. Default 1.
    no_diagnostics : :obj:`bool`
        Disable diagnostic outputs in native format.
    diagnostics_dir : :obj:`str` or :obj:`None`
        Output directory for diagnostics. Defaults to ``<output_stem>_diagnostics``.
    diagnostic_maps : :obj:`list` or :obj:`None`
        Diagnostic maps to write. Supported: ``mean``, ``element_id``, ``n_non_nan``.
    """
    cohort_df = pd.read_csv(cohort_file)
    output_path = Path(output)

    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0
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
    scalars, sources_lists = _load_cohort_voxels(cohort_df, group_mask_matrix, s3_workers)
    maps_to_write = cli_utils.normalize_diagnostic_maps(diagnostic_maps)

    if not no_diagnostics:
        output_diag_dir = (
            Path(diagnostics_dir)
            if diagnostics_dir is not None
            else cli_utils.default_diagnostics_dir(output_path)
        )
        output_diag_dir.mkdir(parents=True, exist_ok=True)
        cli_diagnostics.verify_nifti_element_mapping(group_mask_img, group_mask_matrix)
        for scalar_name, rows in scalars.items():
            diagnostics = cli_diagnostics.summarize_rows(rows)
            cli_diagnostics.write_nifti_diagnostics(
                maps=maps_to_write,
                scalar_name=scalar_name,
                diagnostics=diagnostics,
                group_mask_img=group_mask_img,
                group_mask_matrix=group_mask_matrix,
                output_dir=output_diag_dir,
            )

    if backend == 'hdf5':
        output_path = cli_utils.prepare_output_parent(output_path)
        with h5py.File(output_path, 'w') as h5_file:
            cli_utils.write_table_dataset(h5_file, 'voxels', voxel_table)
            cli_utils.write_hdf5_scalar_matrices(
                h5_file,
                scalars,
                sources_lists,
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                chunk_voxels=chunk_voxels,
                target_chunk_mb=target_chunk_mb,
            )
        return int(not output_path.exists())

    cli_utils.write_tiledb_scalar_matrices(
        output_path,
        scalars,
        sources_lists,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
    )
    return 0


def nifti_to_h5_main(**kwargs):
    """Entry point for the ``modelarrayio nifti-to-h5`` command."""
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    return nifti_to_h5(**kwargs)


def _parse_nifti_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of volume data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    # NIfTI-specific arguments
    parser.add_argument(
        '--group-mask-file',
        '--group_mask_file',
        help='Path to a group mask file',
        required=True,
        type=IsFile,
    )

    # Common arguments
    add_to_modelarray_args(parser, default_output='voxelarray.h5')
    add_diagnostics_args(parser)
    return parser
