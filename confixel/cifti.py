import argparse
import os
from collections import defaultdict
import os.path as op
import numpy as np
import nibabel as nb
import pandas as pd
from tqdm import tqdm
import h5py
from .h5_storage import create_empty_scalar_matrix_dataset
from .parser import add_relative_root_arg, add_output_hdf5_arg, add_cohort_arg, add_storage_args


def extract_cifti_scalar_data(cifti_file, reference_brain_names=None):
    """
    Load a scalar cifti file and get its data and mapping

    Parameters:
    -----------

      cifti_file: pathlike
        CIFTI2 file on disk

      reference_brain_names: np.ndarray
        Array of vertex names
    Returns:
    --------

      cifti_scalar_data: np.ndarray
        The scalar data from the cifti file

      brain_structures: np.ndarray
        The per-greyordinate brain structures as strings

    """

    cifti = nb.load(cifti_file)
    cifti_hdr = cifti.header
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    if len(axes) > 2:
        raise Exception("Only 2 axes should be present in a scalar cifti file")
    if len(axes) < 2:
        raise Exception()

    scalar_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ScalarAxis)]
    brain_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.BrainModelAxis)]

    if not len(scalar_axes) == 1:
        raise Exception(f"Only one scalar axis should be present. Found {scalar_axes}")
    if not len(brain_axes) == 1:
        raise Exception(f"Only one brain axis should be present. Found {brain_axes}")
    brain_axis = brain_axes.pop()


    cifti_data = cifti.get_fdata().squeeze()
    if not cifti_data.ndim == 1:
        raise Exception("Too many dimensions in the cifti data")
    brain_names = brain_axis.name
    if not cifti_data.shape[0] == brain_names.shape[0]:
        raise Exception("Mismatch between the brain names and data array")

    if reference_brain_names is not None:
        if not (brain_names == reference_brain_names).all():
            raise Exception(f"Incosistent vertex names in cifti file {cifti_file}")

    return cifti_data, brain_names


    # vertex_table = pd.DataFrame(
    #     dict(
    #         vertex_id=np.arange(cifti_data.shape[0]),
    #         structure_name=brain_names)


def brain_names_to_dataframe(brain_names):
    # Make a lookup table for greyordinates
    structure_ids, structure_names = pd.factorize(brain_names)
    # Make them a list of strings
    structure_name_strings = list(map(str, structure_names))

    greyordinate_df = pd.DataFrame(
        {"vertex_id": np.arange(structure_ids.shape[0]),
         "structure_id": structure_ids})

    return greyordinate_df, structure_name_strings


def write_hdf5(cohort_file, output_h5='fixeldb.h5', relative_root='/',
               storage_dtype='float32',
               compression='gzip',
               compression_level=4,
               shuffle=True,
               chunk_voxels=0,
               target_chunk_mb=2.0):
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

    # gather cohort data
    cohort_df = pd.read_csv(op.join(relative_root, cohort_file))

    # upload each cohort's data
    scalars = defaultdict(list)
    sources_lists = defaultdict(list)
    last_brain_names = None
    for ix, row in tqdm(cohort_df.iterrows(), total=cohort_df.shape[0]):   # ix: index of row (start from 0); row: one row of data
        scalar_file = op.join(relative_root, row['source_file'])
        cifti_data, brain_names = extract_cifti_scalar_data(
            scalar_file, reference_brain_names=last_brain_names)
        last_brain_names = brain_names.copy()
        scalars[row['scalar_name']].append(cifti_data)   # append to specific scalar_name
        sources_lists[row['scalar_name']].append(row['source_file'])  # append source mif filename to specific scalar_name

    # Write the output
    output_file = op.join(relative_root, output_h5)
    f = h5py.File(output_file, "w")

    greyordinate_table, structure_names = brain_names_to_dataframe(last_brain_names)
    greyordinatesh5 = f.create_dataset(name="greyordinates", data=greyordinate_table.to_numpy().T)
    greyordinatesh5.attrs['column_names'] = list(greyordinate_table.columns)
    greyordinatesh5.attrs['structure_names'] = structure_names

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

        for row_idx, row_data in enumerate(scalars[scalar_name]):
            dset[row_idx, :] = row_data
    f.close()
    return int(not op.exists(output_file))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Create a hdf5 file of CIDTI2 dscalar data")
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
    status = write_hdf5(cohort_file=args.cohort_file,
                        output_h5=args.output_hdf5,
                        relative_root=args.relative_root,
                        storage_dtype=args.dtype,
                        compression=args.compression,
                        compression_level=args.compression_level,
                        shuffle=args.shuffle,
                        chunk_voxels=args.chunk_voxels,
                        target_chunk_mb=args.target_chunk_mb)
    return status


def _h5_to_ciftis(example_cifti, h5_file, analysis_name, cifti_output_dir):
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
    example_cifti: pathlike
        abspath to a scalar cifti file. Its header is used as a template
    h5_file: str
        abspath to an h5 file that contains statistical results and their metadata.
    analysis_name: str
        the name for the analysis results to be saved
    fixel_output_dir: str
        abspath to where the output cifti files will go.
    Outputs
    =======
    None
    """
    # Get a template nifti image.
    cifti = nb.load(example_cifti)
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
    if not op.isdir(cifti_output_dir):
        os.mkdir(cifti_output_dir)

    for result_col, result_name in enumerate(results_names):
        valid_result_name = result_name.replace(" ", "_").replace("/", "_")
        out_cifti = op.join(cifti_output_dir, analysis_name + "_" + valid_result_name + '.nii')
        temp_cifti2 = nb.Cifti2Image(
            results_matrix[result_col, :].reshape(1,-1),
            header=cifti.header,
            nifti_header=cifti.nifti_header)
        temp_cifti2.to_filename(out_cifti)

        # if this result is p.value, also write out 1-p.value (1m.p.value)
        if "p.value" in valid_result_name:   # the result name contains "p.value" (from R package broom)
            valid_result_name_1mpvalue = valid_result_name.replace("p.value", "1m.p.value")
            out_cifti_1mpvalue = op.join(cifti_output_dir, analysis_name + "_" + valid_result_name_1mpvalue + '.mif')
            output_mifvalues_1mpvalue = 1 - results_matrix[result_col, :]   # 1 minus
            temp_nifti2_1mpvalue = nb.Cifti2Image(
                output_mifvalues_1mpvalue.reshape(1, -1),
                header=cifti.header,
                nifti_header=cifti.nifti_header)
            temp_nifti2_1mpvalue.to_filename(out_cifti_1mpvalue)


def h5_to_ciftis():
    parser = get_h5_to_ciftis_parser()
    args = parser.parse_args()

    out_cifti_dir = op.join(args.relative_root, args.output_dir)  # absolute path for output dir

    if op.exists(out_cifti_dir):
        print("WARNING: Output directory exists")
    os.makedirs(out_cifti_dir, exist_ok=True)

    # Get an example mif file
    cohort_df = pd.read_csv(op.join(args.relative_root, args.cohort_file))
    example_cifti = op.join(args.relative_root, cohort_df['source_file'][0])
    h5_input = op.join(args.relative_root, args.input_hdf5)
    analysis_name = args.analysis_name
    _h5_to_ciftis(example_cifti, h5_input, analysis_name, out_cifti_dir)


def get_h5_to_ciftis_parser():
    parser = argparse.ArgumentParser(
        description="Create a directory with cifti results from an hdf5 file")
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
        help="Name for the statistical analysis results to be saved.")
    parser.add_argument(
        "--input-hdf5", "--input_hdf5",
        help="Name of HDF5 (.h5) file where results outputs are saved.")
    parser.add_argument(
        "--output-dir", "--output_dir",
        help="Fixel directory where outputs will be saved. If the directory does not exist, it will be automatically created.")
    return parser


if __name__ == "__main__":
    main()
