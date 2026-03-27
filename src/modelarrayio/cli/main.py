"""ModelArrayIO command-line interface."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version

from modelarrayio.cli.cifti_to_h5 import _parse_cifti_to_h5, cifti_to_h5_main
from modelarrayio.cli.h5_to_cifti import _parse_h5_to_cifti, h5_to_cifti_main
from modelarrayio.cli.h5_to_mif import _parse_h5_to_mif, h5_to_mif_main
from modelarrayio.cli.h5_to_nifti import _parse_h5_to_nifti, h5_to_nifti_main
from modelarrayio.cli.mif_to_h5 import _parse_mif_to_h5, mif_to_h5_main
from modelarrayio.cli.nifti_to_h5 import _parse_nifti_to_h5, nifti_to_h5_main

COMMANDS = [
    ('mif-to-h5', _parse_mif_to_h5, mif_to_h5_main),
    ('nifti-to-h5', _parse_nifti_to_h5, nifti_to_h5_main),
    ('cifti-to-h5', _parse_cifti_to_h5, cifti_to_h5_main),
    ('h5-to-mif', _parse_h5_to_mif, h5_to_mif_main),
    ('h5-to-nifti', _parse_h5_to_nifti, h5_to_nifti_main),
    ('h5-to-cifti', _parse_h5_to_cifti, h5_to_cifti_main),
]


def _get_version() -> str:
    try:
        from modelarrayio.__about__ import __version__
    except ImportError:
        try:
            return version('modelarrayio')
        except PackageNotFoundError:
            return '0+unknown'
    return __version__


def _get_parser():
    parser = argparse.ArgumentParser(prog='modelarrayio', allow_abbrev=False)
    parser.add_argument(
        '-V', '--version', action='version', version=f'modelarrayio {_get_version()}'
    )
    subparsers = parser.add_subparsers(help='modelarrayio subcommands')

    for command, parser_func, run_func in COMMANDS:
        subparser = parser_func()
        subparser.set_defaults(func=run_func)
        subparsers.add_parser(
            command,
            parents=[subparser],
            help=subparser.description,
            add_help=False,
            allow_abbrev=False,
        )

    return parser


def main(argv=None):
    """Entry point for the ``modelarrayio`` command."""
    parser = _get_parser()
    options = parser.parse_args(argv)
    if not hasattr(options, 'func'):
        parser.print_help()
        return 1
    args = vars(options).copy()
    args.pop('func')
    return options.func(**args)
