"""Export statistical results from an HDF5 modelarray file to neuroimaging formats."""

from __future__ import annotations

import argparse
import logging
import shutil
from functools import partial
from pathlib import Path

import pandas as pd

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.h5_to_cifti import h5_to_cifti
from modelarrayio.cli.h5_to_mif import h5_to_mif
from modelarrayio.cli.h5_to_nifti import h5_to_nifti
from modelarrayio.cli.parser_utils import _is_file, add_log_level_arg

logger = logging.getLogger(__name__)


def export_results(
    in_file,
    analysis_name,
    output_dir,
    group_mask_file=None,
    output_extension='.nii.gz',
    index_file=None,
    directions_file=None,
    cohort_file=None,
    example_file=None,
):
    """Export statistical results from an HDF5 modelarray file.

    The modality is inferred from the arguments provided:

    * **NIfTI**: ``group_mask_file`` is given
    * **MIF/fixel**: ``index_file`` and ``directions_file`` are given
    * **CIFTI**: only ``cohort_file`` or ``example_file`` is given

    Parameters
    ----------
    in_file : path-like
        HDF5 file containing statistical results.
    analysis_name : str
        Name of the statistical analysis results group inside the HDF5 file.
    output_dir : path-like
        Directory where output files will be written.
    group_mask_file : path-like, optional
        NIfTI binary group mask. Required for NIfTI results.
    output_extension : str, optional
        NIfTI output extension: ``'.nii.gz'`` (default) or ``'.nii'``.
    index_file : path-like, optional
        Nifti2 index file. Required for MIF/fixel results.
    directions_file : path-like, optional
        Nifti2 directions file. Required for MIF/fixel results.
    cohort_file : path-like, optional
        CSV cohort file used to locate an example source file.
        Required for CIFTI or MIF if ``example_file`` is not given.
    example_file : path-like, optional
        Path to an example source file whose header serves as a template.
        Required for CIFTI or MIF if ``cohort_file`` is not given.
    """
    if group_mask_file is not None:
        modality = 'nifti'
    elif index_file is not None or directions_file is not None:
        modality = 'mif'
    elif cohort_file is not None or example_file is not None:
        modality = 'cifti'
    else:
        raise ValueError(
            'Cannot determine modality. Provide --group-mask-file (NIfTI), '
            '--index-file/--directions-file (MIF), or --cohort-file/--example-file (CIFTI).'
        )
    logger.info('Detected modality: %s', modality)

    output_path = cli_utils.prepare_output_directory(output_dir, logger)

    if modality == 'nifti':
        h5_to_nifti(
            in_file=in_file,
            analysis_name=analysis_name,
            group_mask_file=group_mask_file,
            output_extension=output_extension,
            output_dir=output_path,
        )
        return 0

    if modality == 'mif':
        if index_file is None or directions_file is None:
            raise ValueError(
                'Both --index-file and --directions-file are required for MIF results.'
            )
        if cohort_file is None and example_file is None:
            raise ValueError('One of --cohort-file or --example-file is required for MIF results.')
        shutil.copyfile(index_file, output_path / Path(index_file).name)
        shutil.copyfile(directions_file, output_path / Path(directions_file).name)
        if example_file is None:
            logger.warning('No example MIF file provided; using first source_file from cohort.')
            example_file = pd.read_csv(cohort_file)['source_file'].iloc[0]
        h5_to_mif(
            example_mif=example_file,
            in_file=in_file,
            analysis_name=analysis_name,
            output_dir=output_path,
        )
        return 0

    # cifti
    if cohort_file is None and example_file is None:
        raise ValueError('One of --cohort-file or --example-file is required for CIFTI results.')
    if example_file is None:
        logger.warning('No example CIFTI file provided; using first source_file from cohort.')
        example_file = pd.read_csv(cohort_file)['source_file'].iloc[0]
    h5_to_cifti(
        example_cifti=example_file,
        in_file=in_file,
        analysis_name=analysis_name,
        output_dir=output_path,
    )
    return 0


def export_results_main(**kwargs):
    """Entry point for the ``modelarrayio export-results`` command."""
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    return export_results(**kwargs)


def _parse_export_results():
    parser = argparse.ArgumentParser(
        description=(
            'Export statistical results from an HDF5 modelarray file to '
            'neuroimaging format files. The modality (NIfTI, CIFTI, or MIF/fixel) '
            'is inferred from the arguments provided.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

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

    nifti_group = parser.add_argument_group('NIfTI arguments (required for NIfTI results)')
    nifti_group.add_argument(
        '--group-mask-file',
        '--group_mask_file',
        help='Path to the NIfTI binary group mask file used during data preparation.',
        type=IsFile,
        default=None,
    )
    nifti_group.add_argument(
        '--output-ext',
        '--output_ext',
        dest='output_extension',
        help='Extension for output NIfTI files.',
        choices=['.nii.gz', '.nii'],
        default='.nii.gz',
    )

    mif_group = parser.add_argument_group('MIF/fixel arguments (required for MIF/fixel results)')
    mif_group.add_argument(
        '--index-file',
        '--index_file',
        help='Nifti2 index file used to reconstruct MIF files.',
        type=IsFile,
        default=None,
    )
    mif_group.add_argument(
        '--directions-file',
        '--directions_file',
        help='Nifti2 directions file used to reconstruct MIF files.',
        type=IsFile,
        default=None,
    )

    template_group = parser.add_argument_group(
        'Template arguments (required for CIFTI and MIF/fixel results)'
    )
    template_source = template_group.add_mutually_exclusive_group()
    template_source.add_argument(
        '--cohort-file',
        '--cohort_file',
        help=(
            'Path to a CSV cohort file. The first source file entry is used as a header template.'
        ),
        type=IsFile,
        default=None,
    )
    template_source.add_argument(
        '--example-file',
        '--example_file',
        help='Path to an example source file whose header is used as a template.',
        type=IsFile,
        default=None,
    )

    add_log_level_arg(parser)
    return parser
