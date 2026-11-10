"""Convert CIFTI2 dscalar data to an HDF5 file."""

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
from modelarrayio.utils.cifti import (
    _get_cifti_parcel_info,
    brain_names_to_dataframe,
    load_cohort_cifti,
)
from modelarrayio.utils.misc import build_scalar_sources
from modelarrayio.utils.s3_utils import configure_s3_client_pool

logger = logging.getLogger(__name__)


def cifti_to_h5(
    cohort_long,
    backend='hdf5',
    output=Path('greyordinatearray.h5'),
    storage_dtype='float32',
    compression=None,
    compression_level=None,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=None,
    s3_workers=None,
    split_outputs=False,
    timing_jsonl=None,
):
    """Load all CIFTI data and write to an HDF5 or TileDB file.

    Parameters
    ----------
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
    compression : :obj:`str`, optional
        Compression filter. ``None`` selects the backend default (``gzip`` for HDF5,
        ``zstd`` for TileDB). ``gzip`` works for both backends; ``lzf`` is HDF5-only;
        ``zstd`` is TileDB-only; ``none`` disables compression.
    compression_level : :obj:`int`, optional
        Compression level (codec-dependent). ``None`` selects the codec default
        (gzip 4, zstd 5).
    shuffle : :obj:`bool`
        Enable shuffle filter
    chunk_voxels : :obj:`int`
        Chunk/tile size along the greyordinate axis (0 = auto)
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
    scalar_sources = build_scalar_sources(cohort_long)
    if not scalar_sources:
        raise ValueError('Unable to derive scalar sources from cohort file.')
    scalar_names = list(scalar_sources)
    worker_count = resolve_loader_workers(workers, s3_workers)
    configure_s3_client_pool(worker_count)
    compression, compression_level = storage_utils.resolve_backend_compression(
        backend, compression, compression_level
    )
    planned_outputs = cli_utils.conversion_output_paths(
        output,
        scalar_names,
        split_outputs=split_outputs,
    )
    validate_timing_jsonl_path(
        timing_jsonl,
        planned_outputs,
        directory_outputs=backend == 'tiledb',
    )
    recorder = TimingRecorder(
        timing_jsonl,
        modality='cifti',
        backend=backend,
        workers=worker_count,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
    )

    scalars, reference_brain_names = load_cohort_cifti(
        cohort_long,
        workers=worker_count,
        storage_dtype=storage_dtype,
    )
    # The loader hands back the decoded first image so the metadata can be
    # derived without reopening the file.
    cifti_type, parcel_arrays = _get_cifti_parcel_info(scalars.reference_image)

    if cifti_type == 'dscalar':
        greyordinate_table, structure_names = brain_names_to_dataframe(reference_brain_names)

        def _hdf5_metadata(h5_file):
            cli_utils.write_table_dataset(
                h5_file,
                'greyordinates',
                greyordinate_table,
                extra_attrs={'structure_names': structure_names},
            )

        hdf5_required_paths = ('greyordinates',)
    else:

        def _hdf5_metadata(h5_file):
            cli_utils.write_hdf5_parcel_arrays(h5_file, parcel_arrays)

        hdf5_required_paths = tuple(f'parcels/{name}' for name in parcel_arrays)

    def _tiledb_metadata(root):
        if parcel_arrays:
            cli_utils.write_tiledb_parcel_arrays(root, parcel_arrays)

    publish_scalar_outputs(
        recorder,
        backend=backend,
        output=output,
        split_outputs=split_outputs,
        scalar_names=scalar_names,
        scalars=scalars,
        sources_by_scalar=scalar_sources,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
        expected_elements=len(reference_brain_names),
        hdf5_metadata=_hdf5_metadata,
        hdf5_required_paths=hdf5_required_paths,
        tiledb_metadata=_tiledb_metadata,
        tiledb_required_paths=tuple(f'parcels/{name}' for name in parcel_arrays),
        column_name_arrays=True,
    )
    return 0
