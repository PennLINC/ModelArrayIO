import argparse
import os
import os.path as op
from collections import defaultdict
import nibabel as nb
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
from .h5_storage import create_empty_scalar_matrix_dataset
from .parser import add_relative_root_arg, add_output_hdf5_arg, add_cohort_arg, add_storage_args



def flattened_image(scalar_image, scalar_mask, group_mask_matrix):
    scalar_mask_img = nb.load(scalar_mask)
    scalar_mask_matrix = scalar_mask_img.get_fdata() > 0
    
    scalar_img = nb.load(scalar_image)
    scalar_matrix = scalar_img.get_fdata()

    scalar_matrix[np.logical_not(scalar_mask_matrix)] = np.nan
    return scalar_matrix[group_mask_matrix].squeeze()     # .shape = (#voxels,)  # squeeze() is to remove the 2nd dimension which is not necessary
    

def h5_to_volumes(h5_file, analysis_name, group_mask_file, output_extension, volume_output_dir):
    """ Convert stat results in .h5 file to a list of volume (.nii or .nii.gz) files
    """

    data_type_tosave = np.float32

    # group-level mask:
    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0
    
    # modify the header:
    header_tosave = group_mask_img.header
    header_tosave.set_data_dtype(data_type_tosave)   # modify the data type (mask's data type could be uint8...)

    # results in .h5 file:
    h5_data = h5py.File(h5_file, "r")
    results_matrix = h5_data['results/' + analysis_name + '/results_matrix']
    names_data = results_matrix.attrs['colnames']  # NOTE: results_matrix: need to be transposed...
    # print(results_matrix.shape)   

    try:
        results_names = names_data.tolist()
    except Exception:
        print("Unable to read column names, using 'componentNNN' instead")
        results_names = ['component%03d' % (n + 1) for n in
                         range(results_matrix.shape[0])]

    # # Make output directory if it does not exist  # has been done in h5_to_volumes_wrapper()
    # if op.isdir(volume_output_dir) == False:
    #     os.mkdir(volume_output_dir)

    # for loop: save stat metric results one by one:
    for result_col, result_name in enumerate(results_names):
        valid_result_name = result_name.replace(" ", "_").replace("/", "_")

        out_file = op.join(volume_output_dir, analysis_name + "_" + valid_result_name + output_extension)
        output = np.zeros(group_mask_matrix.shape)
        data_tosave = results_matrix[result_col, :]
        data_tosave = data_tosave.astype(data_type_tosave)  # make sure each result image's data type is the correct one
        output[group_mask_matrix] = data_tosave
        output_img = nb.Nifti1Image(output, affine=group_mask_img.affine,
                                    header=header_tosave)
        output_img.to_filename(out_file)

        # if this result is p.value, also write out 1-p.value (1m.p.value)
        if "p.value" in valid_result_name:   # the result name contains "p.value" (from R package broom)
            valid_result_name_1mpvalue = valid_result_name.replace("p.value", "1m.p.value")
            out_file_1mpvalue = op.join(volume_output_dir, analysis_name + "_" + valid_result_name_1mpvalue + output_extension)
            output_1mpvalue = np.zeros(group_mask_matrix.shape)
            data_tosave = 1 - results_matrix[result_col, :]    # 1 minus
            data_tosave = data_tosave.astype(data_type_tosave)  # make sure each result image's data type is the correct one
            output_1mpvalue[group_mask_matrix] = data_tosave 
            output_img_1mpvalue = nb.Nifti1Image(output_1mpvalue, affine=group_mask_img.affine,
                                                header=header_tosave)
            output_img_1mpvalue.to_filename(out_file_1mpvalue)



def h5_to_volumes_wrapper():
    parser = get_h5_to_volume_parser()
    args = parser.parse_args()

    volume_output_dir = op.join(args.relative_root, args.output_dir)  # absolute path for output dir
    
    if op.exists(volume_output_dir):
        print("WARNING: Output directory exists")
    os.makedirs(volume_output_dir, exist_ok=True)

    # any files to copy?

    # other arguments:
    group_mask_file = op.join(args.relative_root, args.group_mask_file)
    h5_input = op.join(args.relative_root, args.input_hdf5)
    analysis_name = args.analysis_name
    output_extension = args.output_ext

    # call function:
    h5_to_volumes(h5_input, analysis_name, group_mask_file, output_extension, volume_output_dir)


def write_hdf5(group_mask_file, cohort_file, 
               output_h5='voxeldb.h5',
               relative_root='/',
               storage_dtype='float32',
               compression='gzip',
               compression_level=4,
               shuffle=True,
               chunk_voxels=0,
               target_chunk_mb=2.0):
    """
    Load all volume data and write to an HDF5 file with configurable storage.
    Parameters
    -----------
    group_mask_file: str
        Path to a NIfTI-1 binary group mask file.
    cohort_file: str
        Path to a CSV with demographic info and paths to data.
    output_h5: str
        Path to a new .h5 file to be written.
    relative_root: str
        Path to which group_mask_file and cohort_file (and its contents) are relative.
    storage_dtype: str
        Floating type to store values. Options: 'float32' (default), 'float64'.
    compression: str
        HDF5 compression filter. Options: 'gzip' (default), 'lzf', 'none'.
    compression_level: int
        Gzip compression level (0-9). Only used when compression == 'gzip'. Default 4.
    shuffle: bool
        Enable HDF5 shuffle filter to improve compression. Default True (effective when compression != 'none').
    chunk_voxels: int
        Chunk size along the voxel axis. If 0, auto-compute using target_chunk_mb. Default 0.
    target_chunk_mb: float
        Target chunk size in MiB when auto-computing chunk_voxels. Default 2.0.
    """
    # gather cohort data
    cohort_df = pd.read_csv(op.join(relative_root, cohort_file))

    # Load the group mask image to define the rows of the matrix
    group_mask_img = nb.load(op.join(relative_root, group_mask_file))
    group_mask_matrix = group_mask_img.get_fdata() > 0     # get_fdata(): get matrix data in float format
    voxel_coords = np.column_stack(np.nonzero(group_mask_img.get_fdata()))  # np.nonzero() returns the coords of nonzero elements; then np.column_stack() stack them together as an (#voxels, 3) array

    # voxel_table: records the coordinations of the nonzero voxels; coord starts from 0 (because using python)
    voxel_table = pd.DataFrame(
        dict(
            voxel_id=np.arange(voxel_coords.shape[0]),
            i=voxel_coords[:, 0],
            j=voxel_coords[:, 1],
            k=voxel_coords[:, 2]))


    # upload each cohort's data
    scalars = defaultdict(list)
    sources_lists = defaultdict(list)
    print("Extracting NIfTI data...")
    for ix, row in tqdm(cohort_df.iterrows(), total=cohort_df.shape[0]):   # ix: index of row (start from 0); row: one row of data
        scalar_file = op.join(relative_root, row['source_file'])
        scalar_mask_file = op.join(relative_root, row['source_mask_file'])
        scalar_data = flattened_image(scalar_file, scalar_mask_file, group_mask_matrix)
        scalars[row['scalar_name']].append(scalar_data)   # append to specific scalar_name
        sources_lists[row['scalar_name']].append(row['source_file'])  # append source mif filename to specific scalar_name

    # Write the output:
    output_file = op.join(relative_root, output_h5)
    # make dir if not exist:
    output_dir = op.dirname(output_file)
    if not op.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # initialize HDF5 file:
    f = h5py.File(output_file, "w")

    voxelsh5 = f.create_dataset(name="voxels", data=voxel_table.to_numpy().T)
    voxelsh5.attrs['column_names'] = list(voxel_table.columns)

    # Storage handled by utility during dataset creation

    for scalar_name in scalars.keys():
        num_subjects = len(scalars[scalar_name])
        num_voxels = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
        dset = create_empty_scalar_matrix_dataset(
            f,
            'scalars/{}/values'.format(scalar_name),
            num_subjects,
            num_voxels,
            storage_dtype=storage_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            chunk_voxels=chunk_voxels,
            target_chunk_mb=target_chunk_mb,
            sources_list=sources_lists[scalar_name])

        for row_idx, row_data in enumerate(scalars[scalar_name]):
            dset[row_idx, :] = row_data

    f.close()
    return int(not op.exists(output_file))

def get_h5_to_volume_parser():
    parser = argparse.ArgumentParser(
        description="Convert statistical results from an hdf5 file to a volume data (NIfTI file)")
    parser.add_argument(
        "--group-mask-file", "--group_mask_file",
        help="Path to a group mask file",
        required=True)
    parser.add_argument(
        "--cohort-file", "--cohort_file",
        help="Path to a csv with demographic info and paths to data.",
        required=True)
    add_relative_root_arg(parser)
    parser.add_argument(
        "--analysis-name", "--analysis_name",
        help="Name of the statistical analysis results to be saved.")
    parser.add_argument(
        "--input-hdf5", "--input_hdf5",
        help="Name of HDF5 (.h5) file where results outputs are saved.")
    parser.add_argument(
        "--output-dir", "--output_dir",
        help="A directory where output volume files will be saved. If the directory does not exist, it will be automatically created.")
    parser.add_argument(
        "--output-ext", "--output_ext",
        help="The extension for output volume data. Options are .nii.gz (default) and .nii. Please provide the prefix dot.",
        default=".nii.gz")
    return parser
    

def get_parser():

    parser = argparse.ArgumentParser(
        description="Create a hdf5 file of volume data")
    parser.add_argument(
        "--group-mask-file", "--group_mask_file",
        help="Path to a group mask file",
        required=True)
    add_cohort_arg(parser)
    add_relative_root_arg(parser)
    add_output_hdf5_arg(parser, default_name="fixelarray.h5")
    add_storage_args(parser)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    import logging
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format='[%(levelname)s] %(name)s: %(message)s')
    status = write_hdf5(group_mask_file=args.group_mask_file, 
                        cohort_file=args.cohort_file, 
                        output_h5=args.output_hdf5,
                        relative_root=args.relative_root,
                        storage_dtype=args.dtype,
                        compression=args.compression,
                        compression_level=args.compression_level,
                        shuffle=args.shuffle,
                        chunk_voxels=args.chunk_voxels,
                        target_chunk_mb=args.target_chunk_mb)


    return status

if __name__ == "__main__":
    main()