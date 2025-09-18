import os.path as op


def add_relative_root_arg(parser):
    parser.add_argument(
        "--relative-root", "--relative_root",
        help=(
            "Root to which all paths are relative, i.e. defining the (absolute) path to "
            "root directory of inputs and outputs."
        ),
        type=op.abspath,
        default="/inputs/")
    return parser


def add_output_hdf5_arg(parser, default_name="fixelarray.h5"):
    parser.add_argument(
        "--output-hdf5", "--output_hdf5",
        help="Name of HDF5 (.h5) file where outputs will be saved.",
        default=default_name)
    return parser


def add_cohort_arg(parser):
    parser.add_argument(
        "--cohort-file", "--cohort_file",
        help="Path to a csv with demographic info and paths to data.",
        required=True)
    return parser


def add_storage_args(parser):
    parser.add_argument(
        "--dtype",
        help="Floating dtype for storing values: float32 (default) or float64",
        choices=["float32", "float64"],
        default="float32")
    parser.add_argument(
        "--compression",
        help="HDF5 compression filter: gzip (default), lzf, none",
        choices=["gzip", "lzf", "none"],
        default="gzip")
    parser.add_argument(
        "--compression-level", "--compression_level",
        type=int,
        help="Gzip compression level 0-9 (only if --compression=gzip). Default 4",
        default=4)
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable HDF5 shuffle filter (enabled by default if compression is used).")
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        "--chunk-voxels", "--chunk_voxels",
        type=int,
        help=(
            "Chunk size along voxel/greyordinate/fixel axis. If 0, auto-compute based on "
            "--target-chunk-mb and number of subjects"
        ),
        default=0)
    parser.add_argument(
        "--target-chunk-mb", "--target_chunk_mb",
        type=float,
        help="Target chunk size in MiB when auto-computing item chunk length. Default 2.0",
        default=2.0)
    parser.add_argument(
        "--log-level", "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default INFO; set to WARNING to reduce verbosity)",
        default="INFO")
    return parser


