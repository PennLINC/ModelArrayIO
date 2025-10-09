import argparse
import shutil
import os
from collections import defaultdict
import os.path as op
import tempfile
import subprocess
import numpy as np
import nibabel as nb
import pandas as pd
from tqdm import tqdm
import h5py
from .h5_storage import create_empty_scalar_matrix_dataset, write_rows_in_column_stripes
from .tiledb_storage import (
    create_empty_scalar_matrix_array as tdb_create_empty,
    write_rows_in_column_stripes as tdb_write_stripes,
)

def find_mrconvert():
    program = 'mrconvert'

    def is_exe(fpath):
        return op.exists(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = op.join(path, program)
        if is_exe(exe_file):
            return program
    return None


def mif_to_nifti2(mif_file):

    if not mif_file.endswith(".nii"):
        dirpath = tempfile.mkdtemp()
        mrconvert = find_mrconvert()
        if mrconvert is None:
            raise Exception("The mrconvert executable could not be found on $PATH")
        nii_file = op.join(dirpath, 'mif.nii')
        proc = subprocess.Popen([mrconvert, mif_file, nii_file], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        _, err = proc.communicate()
    else:
        nii_file = mif_file
        dirpath = None
    if not op.exists(nii_file):
        raise Exception(err)
    nifti2_img = nb.load(nii_file)
    data = nifti2_img.get_fdata(dtype=np.float32).squeeze()
    # ... do stuff with dirpath
    if dirpath:
        shutil.rmtree(dirpath)
    return nifti2_img, data


def nifti2_to_mif(nifti2_image, mif_file):
# Note: because -force is not turned on in "mrconvert", the output files won't be overwritten!


    mrconvert = find_mrconvert()
    if mrconvert is None:
        raise Exception("The mrconvert executable could not be found on $PATH")

    nii_file = mif_file.replace('.mif','.nii')
    nifti2_image.to_filename(nii_file)  # save as .nii first

    # convert .nii to .mif
    proc = subprocess.Popen([mrconvert, nii_file, mif_file], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    _, err = proc.communicate()

    if not op.exists(mif_file):
        raise Exception(err)
        
    os.remove(nii_file)   # remove temporary .nii file

def gather_fixels(index_file, directions_file):
    """
    Load the index and directions files to get lookup tables.
    Parameters
    -----------
    index_file: str
        path to a Nifti2 index file
    directions_file: str
        path to a Nifti2 directions file
    """

    index_img, index_data = mif_to_nifti2(index_file)
    count_vol = index_data[..., 0].astype(np.uint32)   # number of fixels in each voxel; by index.mif definition
    id_vol = index_data[..., 1]  # index of the first fixel in this voxel, in the list of all fixels (in directions.mif, FD.mif, etc)
    max_id = id_vol.max()
    max_fixel_id = max_id + int(count_vol[id_vol == max_id])  # = the maximum id of fixels + 1 = # of fixels in entire image
    voxel_mask = count_vol > 0   # voxels that contains fixel(s), =1
    masked_ids = id_vol[voxel_mask]  # 1D array, len = # of voxels with fixel(s), value see id_vol
    masked_counts = count_vol[voxel_mask]  # dim as masked_ids; value see count_vol
    id_sort = np.argsort(masked_ids)   #  indices that would sort array masked_ids value (i.e. first fixel's id in this voxel) from lowest to highest; so it's sorting voxels by their first fixel id
    sorted_counts = masked_counts[id_sort]
    voxel_coords = np.column_stack(np.nonzero(count_vol))  # dim: [# of voxels with fixel(s)] x 3, each row is the subscript i.e. (i,j,k) in 3D image of a voxel with fixel

    fixel_id = 0
    fixel_ids = np.arange(max_fixel_id, dtype=np.int32)
    fixel_voxel_ids = np.zeros_like(fixel_ids)
    for voxel_id, fixel_count in enumerate(sorted_counts):
        for _ in range(fixel_count):
            fixel_voxel_ids[fixel_id] = voxel_id   # fixel_voxel_ids: 1D, len = # of fixels; each value is the voxel_id of the voxel where this fixel locates
            fixel_id += 1
    sorted_coords = voxel_coords[id_sort]

    voxel_table = pd.DataFrame(
        dict(
            voxel_id=np.arange(voxel_coords.shape[0]),
            i=sorted_coords[:, 0],
            j=sorted_coords[:, 1],
            k=sorted_coords[:, 2]))

    directions_img, directions_data = mif_to_nifti2(directions_file)
    fixel_table = pd.DataFrame(
        dict(
            fixel_id=fixel_ids,
            voxel_id=fixel_voxel_ids,
            x=directions_data[:,0],
            y=directions_data[:,1],
            z=directions_data[:,2])
        )

    return fixel_table, voxel_table


def write_storage(index_file, directions_file, cohort_file,
                  backend='hdf5',
                  output_h5='fixeldb.h5',
                  output_tdb='arraydb.tdb',
                  relative_root='/',
                  storage_dtype='float32',
                  compression='gzip',
                  compression_level=4,
                  shuffle=True,
                  chunk_voxels=0,
                  target_chunk_mb=2.0,
                  tdb_compression='zstd',
                  tdb_compression_level=5,
                  tdb_shuffle=True,
                  tdb_tile_voxels=0,
                  tdb_target_tile_mb=2.0):
    """
    Load all fixeldb data.
    Parameters
    -----------
    index_file: str
        path to a Nifti2 index file
    directions_file: str
        path to a Nifti2 directions file
    cohort_file: str
        path to a csv with demographic info and paths to data
    output_h5: str
        path to a new .h5 file to be written
    relative_root: str
        path to which index_file, directions_file and cohort_file (and its contents) are relative
    """
    # gather fixel data
    fixel_table, voxel_table = gather_fixels(op.join(relative_root, index_file),
                                             op.join(relative_root, directions_file))

    # gather cohort data
    cohort_df = pd.read_csv(op.join(relative_root, cohort_file))

    # upload each cohort's data
    scalars = defaultdict(list)
    sources_lists = defaultdict(list)
    print("Extracting .mif data...")
    for ix, row in tqdm(cohort_df.iterrows(), total=cohort_df.shape[0]):   # ix: index of row (start from 0); row: one row of data
        scalar_file = op.join(relative_root, row['source_file'])
        scalar_img, scalar_data = mif_to_nifti2(scalar_file)
        scalars[row['scalar_name']].append(scalar_data)   # append to specific scalar_name
        sources_lists[row['scalar_name']].append(row['source_file'])  # append source mif filename to specific scalar_name

    # Write the output
    if backend == 'hdf5':
        output_file = op.join(relative_root, output_h5)
        f = h5py.File(output_file, "w")

        fixelsh5 = f.create_dataset(name="fixels", data=fixel_table.to_numpy().T)
        fixelsh5.attrs['column_names'] = list(fixel_table.columns)

        voxelsh5 = f.create_dataset(name="voxels", data=voxel_table.to_numpy().T)
        voxelsh5.attrs['column_names'] = list(voxel_table.columns)

        for scalar_name in scalars.keys():
            num_subjects = len(scalars[scalar_name])
            num_items = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
            dset = create_empty_scalar_matrix_dataset(
                f,
                'scalars/{}/values'.format(scalar_name),
                num_subjects,
                num_items,
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                chunk_voxels=chunk_voxels,
                target_chunk_mb=target_chunk_mb,
                sources_list=sources_lists[scalar_name])

            write_rows_in_column_stripes(dset, scalars[scalar_name])
        f.close()
        return int(not op.exists(output_file))
    else:
        base_uri = op.join(relative_root, output_tdb)
        os.makedirs(base_uri, exist_ok=True)
        for scalar_name in scalars.keys():
            num_subjects = len(scalars[scalar_name])
            num_items = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
            dataset_path = f'scalars/{scalar_name}/values'
            tdb_create_empty(
                base_uri,
                dataset_path,
                num_subjects,
                num_items,
                storage_dtype=storage_dtype,
                compression=tdb_compression,
                compression_level=tdb_compression_level,
                shuffle=tdb_shuffle,
                tile_voxels=tdb_tile_voxels,
                target_tile_mb=tdb_target_tile_mb,
                sources_list=sources_lists[scalar_name],
            )
            uri = op.join(base_uri, dataset_path)
            tdb_write_stripes(uri, scalars[scalar_name])
        return 0


def get_parser():

    parser = argparse.ArgumentParser(
        description="Create a hdf5 file of fixel data")
    parser.add_argument(
        "--index-file", "--index_file",
        help="Index File",
        required=True)
    parser.add_argument(
        "--directions-file", "--directions_file",
        help="Directions File",
        required=True)
    parser.add_argument(
        "--cohort-file", "--cohort_file",
        help="Path to a csv with demographic info and paths to data.",
        required=True)
    parser.add_argument(
        "--relative-root", "--relative_root",
        help="Root to which all paths are relative, i.e. defining the (absolute) path to root directory of index_file, directions_file, cohort_file, and output_hdf5.",
        type=op.abspath, 
        default="/inputs/")
    parser.add_argument(
        "--output-hdf5", "--output_hdf5",
        help="Name of HDF5 (.h5) file where outputs will be saved.", 
        default="fixelarray.h5")
    parser.add_argument(
        "--output-tiledb", "--output_tiledb",
        help="Base URI (directory) where TileDB arrays will be created.", 
        default="arraydb.tdb")
    parser.add_argument(
        "--backend",
        help="Storage backend for subject-by-element matrix",
        choices=["hdf5", "tiledb"],
        default="hdf5")
    # Storage configuration (match voxels.py)
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
        "--tdb-compression", "--tdb_compression",
        help="TileDB compression: zstd (default), gzip, none",
        choices=["zstd", "gzip", "none"],
        default="zstd")
    parser.add_argument(
        "--compression-level", "--compression_level",
        type=int,
        help="Gzip compression level 0-9 (only if --compression=gzip). Default 4",
        default=4)
    parser.add_argument(
        "--tdb-compression-level", "--tdb_compression_level",
        type=int,
        help="Compression level for TileDB (codec-dependent).",
        default=5)
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable HDF5 shuffle filter (enabled by default if compression is used).")
    parser.add_argument(
        "--tdb-no-shuffle",
        dest="tdb_shuffle",
        action="store_false",
        help="Disable TileDB shuffle filter (enabled by default).")
    parser.set_defaults(shuffle=True)
    parser.set_defaults(tdb_shuffle=True)
    parser.add_argument(
        "--chunk-voxels", "--chunk_voxels",
        type=int,
        help="Chunk size along fixel/voxel axis. If 0, auto-compute based on --target-chunk-mb and number of subjects",
        default=0)
    parser.add_argument(
        "--tdb-tile-voxels", "--tdb_tile_voxels",
        type=int,
        help="Tile length along item axis for TileDB. If 0, auto-compute based on --tdb-target-tile-mb",
        default=0)
    parser.add_argument(
        "--target-chunk-mb", "--target_chunk_mb",
        type=float,
        help="Target chunk size in MiB when auto-computing item chunk length. Default 2.0",
        default=2.0)
    parser.add_argument(
        "--tdb-target-tile-mb", "--tdb_target_tile_mb",
        type=float,
        help="Target tile size in MiB when auto-computing item tile length. Default 2.0",
        default=2.0)
    parser.add_argument(
        "--log-level", "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default INFO; set to WARNING to reduce verbosity)",
        default="INFO")
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()
    import logging
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format='[%(levelname)s] %(name)s: %(message)s')
    status = write_storage(index_file=args.index_file,
                           directions_file=args.directions_file,
                           cohort_file=args.cohort_file,
                           backend=args.backend,
                           output_h5=args.output_hdf5,
                           output_tdb=args.output_tiledb,
                           relative_root=args.relative_root,
                           storage_dtype=args.dtype,
                           compression=args.compression,
                           compression_level=args.compression_level,
                           shuffle=args.shuffle,
                           chunk_voxels=args.chunk_voxels,
                           target_chunk_mb=args.target_chunk_mb,
                           tdb_compression=args.tdb_compression,
                           tdb_compression_level=args.tdb_compression_level,
                           tdb_shuffle=args.tdb_shuffle,
                           tdb_tile_voxels=args.tdb_tile_voxels,
                           tdb_target_tile_mb=args.tdb_target_tile_mb)
    return status


def h5_to_mifs(example_mif, h5_file, analysis_name, fixel_output_dir):
    """Writes the contents of an hdf5 file to a fixels directory.
    The ``h5_file`` parameter should point to an HDF5 file that contains at least two
    datasets. There must be one called ``results/results_matrix``, that contains a
    matrix of fixel results. Each column contains a single result and each row is a
    fixel. This matrix should be of type float. The second required dataset must be
    named ``results/has_names``. This data can be of any type and does not need to contain
    more than a single row of data. Instead, its attributes are read to get column names
    for the data represented in ``results/results_matrix``.
    The function takes the example mif file and converts it to Nifti2 to get a header.
    Then each column in ``results/results_matrix`` is extracted to fill the data of a
    new Nifti2 file that gets converted to mif and named according to the corresponding
    item in ``results/has_names``.
    Parameters
    ==========
    example_mif: str
        abspath to a scalar mif file. Its header is used as a template
    h5_file: str
        abspath to an h5 file that contains statistical results and their metadata.
    analysis_name: str
        the name for the analysis results to be saved
    fixel_output_dir: str
        abspath to where the output fixel data will go. the index and directions mif files
        should already be copied here.
    Outputs
    =======
    None
    """
    # Get a template nifti image.
    nifti2_img, _ = mif_to_nifti2(example_mif)
    h5_data = h5py.File(h5_file, "r")
    results_matrix = h5_data['results/' + analysis_name + '/results_matrix']
    names_data = results_matrix.attrs['colnames']  # NOTE: results_matrix: need to be transposed...
    # print(results_matrix.shape)    

    # print(h5_data['results/' + analysis_name + '/results_matrix'].attrs['column_names'])
    
    try:
        results_names = names_data.tolist()
    except Exception:
        print("Unable to read column names, using 'componentNNN' instead")
        results_names = ['component%03d' % (n + 1) for n in
                         range(results_matrix.shape[0])]

    # Make output directory if it does not exist
    if not op.isdir(fixel_output_dir):
        os.mkdir(fixel_output_dir)
        
    for result_col, result_name in enumerate(results_names):
        valid_result_name = result_name.replace(" ", "_").replace("/", "_")
        out_mif = op.join(fixel_output_dir, analysis_name + "_" + valid_result_name + '.mif')
        temp_nifti2 = nb.Nifti2Image(results_matrix[result_col, :].reshape(-1, 1, 1),
                                     nifti2_img.affine,
                                     header=nifti2_img.header)
        nifti2_to_mif(temp_nifti2, out_mif)

        # if this result is p.value, also write out 1-p.value (1m.p.value)
        if "p.value" in valid_result_name:   # the result name contains "p.value" (from R package broom)
            valid_result_name_1mpvalue = valid_result_name.replace("p.value", "1m.p.value")
            out_mif_1mpvalue = op.join(fixel_output_dir, analysis_name + "_" + valid_result_name_1mpvalue + '.mif')
            output_mifvalues_1mpvalue = 1 - results_matrix[result_col, :]   # 1 minus
            temp_nifti2_1mpvalue = nb.Nifti2Image(output_mifvalues_1mpvalue.reshape(-1, 1, 1),
                                                    nifti2_img.affine,
                                                    header=nifti2_img.header)
            nifti2_to_mif(temp_nifti2_1mpvalue, out_mif_1mpvalue)


def h5_to_fixels():
    parser = get_h5_to_fixels_parser()
    args = parser.parse_args()

    out_fixel_dir = op.join(args.relative_root, args.output_dir)  # absolute path for output dir

    if op.exists(out_fixel_dir):
        print("WARNING: Output directory exists")
    os.makedirs(out_fixel_dir, exist_ok=True)

    # Copy in the index and directions
    shutil.copyfile(op.join(args.relative_root, args.directions_file),
                    op.join(out_fixel_dir, op.split(args.directions_file)[1]))
    shutil.copyfile(op.join(args.relative_root, args.index_file),
                    op.join(out_fixel_dir, op.split(args.index_file)[1]))

    # Get an example mif file
    cohort_df = pd.read_csv(op.join(args.relative_root, args.cohort_file))
    example_mif = op.join(args.relative_root, cohort_df['source_file'][0])
    h5_input = op.join(args.relative_root, args.input_hdf5)
    analysis_name = args.analysis_name
    h5_to_mifs(example_mif, h5_input, analysis_name, out_fixel_dir)



def get_h5_to_fixels_parser():
    parser = argparse.ArgumentParser(
        description="Create a fixel directory from an hdf5 file")
    parser.add_argument(
        "--index-file", "--index_file",
        help="Index File",
        required=True)
    parser.add_argument(
        "--directions-file", "--directions_file",
        help="Directions File",
        required=True)
    parser.add_argument(
        "--cohort-file", "--cohort_file",
        help="Path to a csv with demographic info and paths to data.",
        required=True)
    parser.add_argument(
        "--relative-root", "--relative_root",
        help="Root to which all paths are relative, i.e. defining the (absolute) path to root directory of index_file, directions_file, cohort_file, input_hdf5, and output_dir.",
        type=os.path.abspath)
    parser.add_argument(
        "--analysis-name", "--analysis_name",
        help="Name for the statistical analysis results to be saved."
    )
    parser.add_argument(
        "--input-hdf5", "--input_hdf5",
        help="Name of HDF5 (.h5) file where results outputs are saved.")
    parser.add_argument(
        "--output-dir", "--output_dir",
        help="Fixel directory where outputs will be saved. If the directory does not exist, it will be automatically created.")
    return parser


if __name__ == "__main__":
    main()

