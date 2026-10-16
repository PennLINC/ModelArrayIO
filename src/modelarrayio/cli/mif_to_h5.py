"""Convert MIF data to an HDF5 file."""

from __future__ import annotations

import logging
from pathlib import Path

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.runtime import (
    TimingRecorder,
    publish_scalar_outputs,
    resolve_loader_workers,
    validate_timing_jsonl_path,
)
from modelarrayio.storage import utils as storage_utils
from modelarrayio.utils.mif import gather_fixels, load_cohort_mif
from modelarrayio.utils.s3_utils import configure_s3_client_pool

logger = logging.getLogger(__name__)


def mif_to_h5(
    index_file,
    directions_file,
    cohort_long,
    backend='hdf5',
    output=Path('fixelarray.h5'),
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
    """Load all fixeldb data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    index_file : :obj:`pathlib.Path`
        Path to a Nifti2 index file
    directions_file : :obj:`pathlib.Path`
        Path to a Nifti2 directions file
    cohort_long : :obj:`pandas.DataFrame`
        Normalised long-format cohort dataframe (from
        :func:`~modelarrayio.utils.misc.load_and_normalize_cohort`).
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
        Maximum number of input sources loaded concurrently, for every backend.
    s3_workers : :obj:`int`
        Deprecated alias for ``workers``.
    split_outputs : :obj:`bool`
        If True, write one output file per scalar. Default False.
    timing_jsonl : path-like, optional
        Append structured per-scalar and total runtime records to this file.

    Returns
    -------
    status : :obj:`int`
        0 if successful, 1 if failed.
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
        modality='mif',
        backend=backend,
        workers=worker_count,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
    )
    # gather fixel data
    fixel_table, voxel_table = gather_fixels(index_file, directions_file)

    logger.info('Extracting .mif data...')
    scalars, sources_lists = load_cohort_mif(
        cohort_long,
        workers=worker_count,
        storage_dtype=storage_dtype,
    )
    if not sources_lists:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    def _hdf5_metadata(h5_file):
        cli_utils.write_table_dataset(h5_file, 'fixels', fixel_table)
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
        expected_elements=len(fixel_table),
        hdf5_metadata=_hdf5_metadata,
        hdf5_required_paths=('fixels', 'voxels'),
    )
    return 0
