from functools import partial
from pathlib import Path


def add_output_hdf5_arg(parser, default_name='fixelarray.h5'):
    parser.add_argument(
        '--output-hdf5',
        '--output_hdf5',
        help='Name of HDF5 (.h5) file where outputs will be saved.',
        default=default_name,
    )
    return parser


def add_cohort_arg(parser):
    parser.add_argument(
        '--cohort-file',
        '--cohort_file',
        help='Path to a csv with demographic info and paths to data.',
        required=True,
        type=partial(_is_file, parser=parser),
    )
    return parser


def add_storage_args(parser):
    parser.add_argument(
        '--dtype',
        help='Floating dtype for storing values: float32 (default) or float64',
        choices=['float32', 'float64'],
        default='float32',
        dest='storage_dtype',
    )
    parser.add_argument(
        '--compression',
        help='HDF5 compression filter: gzip (default), lzf, none',
        choices=['gzip', 'lzf', 'none'],
        default='gzip',
    )
    parser.add_argument(
        '--compression-level',
        '--compression_level',
        type=int,
        help='Gzip compression level 0-9 (only if --compression=gzip). Default 4',
        default=4,
    )
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='Disable HDF5 shuffle filter (enabled by default if compression is used).',
        default=True,
    )

    chunk_allocation_group = parser.add_mutually_exclusive_group()
    chunk_allocation_group.add_argument(
        '--chunk-voxels',
        '--chunk_voxels',
        type=int,
        help=(
            'Chunk size along voxel/greyordinate/fixel axis. If 0, auto-compute based on '
            '--target-chunk-mb and number of subjects'
        ),
        default=0,
    )
    chunk_allocation_group.add_argument(
        '--target-chunk-mb',
        '--target_chunk_mb',
        type=float,
        help='Target chunk size in MiB when auto-computing item chunk length. Default 2.0',
        default=2.0,
    )

    parser.add_argument(
        '--log-level',
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default INFO; set to WARNING to reduce verbosity)',
        default='INFO',
    )
    return parser


def add_backend_arg(parser):
    parser.add_argument(
        '--backend',
        help='Storage backend for subject-by-element matrix',
        choices=['hdf5', 'tiledb'],
        default='hdf5',
    )
    return parser


def add_output_tiledb_arg(parser, default_name='arraydb.tdb'):
    parser.add_argument(
        '--output-tiledb',
        '--output_tiledb',
        help=(
            'Base URI (directory) where TileDB arrays will be created. '
            'If relative, it is joined to --relative-root.'
        ),
        default=default_name,
    )
    return parser


def add_tiledb_storage_args(parser):
    parser.add_argument(
        '--tdb-compression',
        '--tdb_compression',
        help='TileDB compression: zstd (default), gzip, none',
        choices=['zstd', 'gzip', 'none'],
        default='zstd',
    )
    parser.add_argument(
        '--tdb-compression-level',
        '--tdb_compression_level',
        type=int,
        help='Compression level for TileDB (codec-dependent).',
        default=5,
    )
    parser.add_argument(
        '--tdb-no-shuffle',
        dest='tdb_shuffle',
        action='store_false',
        help='Disable TileDB shuffle filter (enabled by default).',
        default=True,
    )

    tile_allocation_group = parser.add_mutually_exclusive_group()
    tile_allocation_group.add_argument(
        '--tdb-tile-voxels',
        '--tdb_tile_voxels',
        type=int,
        help=(
            'Tile length along item axis. If 0, auto-compute based on --tdb-target-tile-mb and '
            'number of subjects'
        ),
        default=0,
    )
    tile_allocation_group.add_argument(
        '--tdb-target-tile-mb',
        '--tdb_target_tile_mb',
        type=float,
        help='Target tile size in MiB when auto-computing item tile length. Default 2.0',
        default=2.0,
    )
    return parser


def add_s3_workers_arg(parser):
    parser.add_argument(
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
    return parser


def add_scalar_columns_arg(parser):
    parser.add_argument(
        '--scalar-columns',
        '--scalar_columns',
        nargs='+',
        help=(
            'Column names containing scalar file paths when the cohort table is in wide format. '
            "If omitted, the cohort file must include 'scalar_name' and 'source_file' columns."
        ),
    )
    return parser


def _path_exists(path, parser):
    """Ensure a given path exists."""
    if path is None or not Path(path).exists():
        raise parser.error(f'Path does not exist: <{path}>.')
    return Path(path).absolute()


def _is_file(path, parser):
    """Ensure a given path exists and it is a file."""
    path = _path_exists(path, parser)
    if not path.is_file():
        raise parser.error(f'Path should point to a file (or symlink of file): <{path}>.')
    return path
