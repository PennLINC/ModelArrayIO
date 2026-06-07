"""Convert per-subject ODX fixel data to an HDF5 (or TileDB) ModelArray file.

The ODX analogue of :mod:`modelarrayio.cli.mif_to_h5`. Each cohort row points
to one per-subject ODX (single-column DPF) produced by
``odx combine --per-subject-odx DIR``; the shared group-fixel geometry is read
directly from the first ODX, so no separate index/directions files are needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py

from modelarrayio.cli import utils as cli_utils
from modelarrayio.utils.odx import gather_fixels_from_odx, load_cohort_odx

logger = logging.getLogger(__name__)


def odx_to_h5(
    cohort_long,
    backend='hdf5',
    output=Path('fixelarray.h5'),
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=1,
    s3_workers=1,
    split_outputs=False,
):
    """Load per-subject ODX fixel data and write a ModelArray store.

    Parameters mirror :func:`modelarrayio.cli.mif_to_h5.mif_to_h5` minus the
    ``index_file``/``directions_file`` arguments — ODX carries its own geometry.

    Returns
    -------
    status : :obj:`int`
        0 if successful, non-zero if an expected output was not written.
    """
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any ODX scalar entries.')

    # Every template-space ODX shares the group-fixel geometry; read it once.
    first_source = str(cohort_long.iloc[0]['source_file'])
    fixel_table, voxel_table = gather_fixels_from_odx(first_source)

    logger.info('Extracting ODX data...')
    scalars, sources_lists = load_cohort_odx(cohort_long, s3_workers)
    if not sources_lists:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    scalar_names = list(sources_lists.keys())

    if backend == 'hdf5':
        if split_outputs:
            outputs: list[Path] = []
            for scalar_name in scalar_names:
                scalar_output = cli_utils.prepare_output_parent(
                    cli_utils.prefixed_output_path(output, scalar_name)
                )
                with h5py.File(scalar_output, 'w') as h5_file:
                    cli_utils.write_table_dataset(h5_file, 'fixels', fixel_table)
                    cli_utils.write_table_dataset(h5_file, 'voxels', voxel_table)
                    cli_utils.write_hdf5_scalar_matrices(
                        h5_file,
                        {scalar_name: scalars[scalar_name]},
                        {scalar_name: sources_lists[scalar_name]},
                        storage_dtype=storage_dtype,
                        compression=compression,
                        compression_level=compression_level,
                        shuffle=shuffle,
                        chunk_voxels=chunk_voxels,
                        target_chunk_mb=target_chunk_mb,
                    )
                outputs.append(scalar_output)
            return int(not all(path.exists() for path in outputs))

        output = cli_utils.prepare_output_parent(output)
        with h5py.File(output, 'w') as h5_file:
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
        return int(not output.exists())

    # tiledb backend
    for scalar_name in scalar_names:
        scalar_output = (
            cli_utils.prefixed_output_path(output, scalar_name) if split_outputs else output
        )
        cli_utils.write_tiledb_scalar_matrices(
            scalar_output,
            {scalar_name: scalars[scalar_name]},
            {scalar_name: sources_lists[scalar_name]},
            storage_dtype=storage_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            chunk_voxels=chunk_voxels,
            target_chunk_mb=target_chunk_mb,
        )
    return 0
