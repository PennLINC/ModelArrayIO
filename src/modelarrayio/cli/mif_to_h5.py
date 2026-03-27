"""Convert MIF data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli import diagnostics as cli_diagnostics
from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import _is_file, add_diagnostics_args, add_to_modelarray_args
from modelarrayio.utils.fixels import gather_fixels, mif_to_nifti2

logger = logging.getLogger(__name__)


def mif_to_h5(
    index_file,
    directions_file,
    cohort_file,
    backend='hdf5',
    output=Path('fixelarray.h5'),
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
    """Load all fixeldb data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    index_file : :obj:`pathlib.Path`
        Path to a Nifti2 index file
    directions_file : :obj:`pathlib.Path`
        Path to a Nifti2 directions file
    cohort_file : :obj:`pathlib.Path`
        Path to a csv with demographic info and paths to data
    backend : :obj:`str`
        Backend to use for storage (``'hdf5'`` or ``'tiledb'``)
    output : :obj:`pathlib.Path`
        Output path. For the hdf5 backend, path to an .h5 file;
        for the tiledb backend, path to a .tdb directory.
    storage_dtype : :obj:`str`
        Floating type to store values
    compression : :obj:`str`
        Compression filter. ``gzip`` works for both backends;
        ``lzf`` is HDF5-only; ``zstd`` is TileDB-only.
    compression_level : :obj:`int`
        Compression level (codec-dependent)
    shuffle : :obj:`bool`
        Enable shuffle filter
    chunk_voxels : :obj:`int`
        Chunk/tile size along the fixel axis (0 = auto)
    target_chunk_mb : :obj:`float`
        Target chunk/tile size in MiB when auto-computing the spatial axis length
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

    Returns
    -------
    status : :obj:`int`
        0 if successful, 1 if failed.
    """
    # gather fixel data
    fixel_table, voxel_table = gather_fixels(index_file, directions_file)
    output_path = Path(output)

    # gather cohort data
    cohort_df = pd.read_csv(cohort_file)
    maps_to_write = cli_utils.normalize_diagnostic_maps(diagnostic_maps)

    # upload each cohort's data
    scalars = defaultdict(list)
    sources_lists = defaultdict(list)
    template_nifti2 = None
    logger.info('Extracting .mif data...')
    for row in tqdm(cohort_df.itertuples(index=False), total=cohort_df.shape[0]):
        scalar_file = row.source_file
        scalar_img, scalar_data = mif_to_nifti2(scalar_file)
        if template_nifti2 is None:
            template_nifti2 = scalar_img
        scalars[row.scalar_name].append(scalar_data)
        sources_lists[row.scalar_name].append(row.source_file)

    if not no_diagnostics:
        output_diag_dir = (
            Path(diagnostics_dir)
            if diagnostics_dir is not None
            else cli_utils.default_diagnostics_dir(output_path)
        )
        output_diag_dir.mkdir(parents=True, exist_ok=True)
        for scalar_name, rows in scalars.items():
            cli_diagnostics.verify_mif_element_mapping(template_nifti2, rows[0].shape[0])
            diagnostics = cli_diagnostics.summarize_rows(rows)
            cli_diagnostics.write_mif_diagnostics(
                maps=maps_to_write,
                scalar_name=scalar_name,
                diagnostics=diagnostics,
                template_nifti2=template_nifti2,
                output_dir=output_diag_dir,
            )

    # Write the output
    if backend == 'hdf5':
        output_path = cli_utils.prepare_output_parent(output_path)
        with h5py.File(output_path, 'w') as h5_file:
            cli_utils.write_table_dataset(h5_file, 'fixels', fixel_table)
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


def mif_to_h5_main(**kwargs):
    """Entry point for the ``modelarrayio mif-to-h5`` command."""
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    return mif_to_h5(**kwargs)


def _parse_mif_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of fixel data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    # MIF-specific arguments
    parser.add_argument(
        '--index-file',
        '--index_file',
        help='Index File',
        required=True,
        type=IsFile,
    )
    parser.add_argument(
        '--directions-file',
        '--directions_file',
        help='Directions File',
        required=True,
        type=IsFile,
    )

    # Common arguments
    add_to_modelarray_args(parser, default_output='fixelarray.h5')
    add_diagnostics_args(parser)
    return parser
