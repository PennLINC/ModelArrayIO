from __future__ import annotations

from functools import partial
from pathlib import Path


def add_to_modelarray_args(parser, default_output='output.h5'):
    """Add arguments common to all commands that prepare data for ModelArray."""
    parser.add_argument(
        '--cohort-file',
        '--cohort_file',
        help='Path to a csv with demographic info and paths to data.',
        required=True,
        type=partial(_is_file, parser=parser),
    )
    parser.add_argument(
        '--output',
        help=(
            'Output path. For the hdf5 backend, path to an .h5 file; '
            'for the tiledb backend, path to a .tdb directory.'
        ),
        default=default_output,
        type=Path,
    )
    parser.add_argument(
        '--scalar-columns',
        '--scalar_columns',
        nargs='+',
        help=(
            'Column names containing scalar file paths when the cohort table is in wide format. '
            'If omitted, the cohort file must include "scalar_name" and "source_file" columns.'
        ),
    )
    parser.add_argument(
        '--backend',
        help='Storage backend for subject-by-element matrix',
        choices=['hdf5', 'tiledb'],
        default='hdf5',
    )
    parser.add_argument(
        '--dtype',
        help='Floating dtype for storing values: float32 (default) or float64',
        choices=['float32', 'float64'],
        default='float32',
        dest='storage_dtype',
    )
    parser.add_argument(
        '--compression',
        help=(
            'Compression filter (default gzip). '
            'gzip works for both backends; '
            'lzf is HDF5-only; '
            'zstd is TileDB-only.'
        ),
        choices=['gzip', 'zstd', 'lzf', 'none'],
        default='gzip',
    )
    parser.add_argument(
        '--compression-level',
        '--compression_level',
        type=int,
        help='Compression level (codec-dependent). Default 4.',
        default=4,
    )
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='Disable shuffle filter (enabled by default when compression is used).',
        default=True,
    )

    chunk_allocation_group = parser.add_mutually_exclusive_group()
    chunk_allocation_group.add_argument(
        '--chunk-voxels',
        '--chunk_voxels',
        type=int,
        help=(
            'Chunk/tile size along voxel/greyordinate/fixel axis. '
            'If 0, auto-compute based on --target-chunk-mb and number of subjects.'
        ),
        default=0,
    )
    chunk_allocation_group.add_argument(
        '--target-chunk-mb',
        '--target_chunk_mb',
        type=float,
        help='Target chunk/tile size in MiB when auto-computing the spatial axis length. Default 2.0.',
        default=2.0,
    )

    tiledb_group = parser.add_argument_group('TileDB arguments')
    tiledb_group.add_argument(
        '--workers',
        type=int,
        help=(
            'Maximum number of parallel TileDB write workers. '
            'Default 1. '
            'Has no effect when --backend=hdf5.'
        ),
        default=1,
    )

    s3_group = parser.add_argument_group('S3 arguments')
    s3_group.add_argument(
        '--s3-workers',
        '--s3_workers',
        type=int,
        default=1,
        help=(
            'Number of parallel worker processes for loading image files. '
            'Set > 1 to enable parallel downloads when cohort paths begin with s3://. '
            'Default 1 (serial).'
        ),
    )

    add_log_level_arg(parser)

    return parser


def add_log_level_arg(parser):
    parser.add_argument(
        '--log-level',
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default INFO; set to WARNING to reduce verbosity)',
        default='INFO',
    )
    return parser


def add_from_modelarray_args(parser):
    parser.add_argument(
        '--analysis-name',
        '--analysis_name',
        help='Name for the statistical analysis results to be saved.',
        required=True,
    )
    parser.add_argument(
        '--input-hdf5',
        '--input_hdf5',
        help='Name of HDF5 (.h5) file where results outputs are saved.',
        type=partial(_is_file, parser=parser),
        dest='in_file',
        required=True,
    )
    parser.add_argument(
        '--output-dir',
        '--output_dir',
        help=(
            'Directory where outputs will be saved. '
            'If the directory does not exist, it will be automatically created.'
        ),
        required=True,
    )

    return parser


def _path_exists(path: str | Path | None, parser) -> Path:
    """Ensure a given path exists."""
    if path is None or not Path(path).exists():
        raise parser.error(f'Path does not exist: <{path}>.')
    return Path(path).absolute()


def _is_file(path: str | Path | None, parser) -> Path:
    """Ensure a given path exists and it is a file."""
    path = _path_exists(path, parser)
    if not path.is_file():
        raise parser.error(f'Path should point to a file (or symlink of file): <{path}>.')
    return path
