"""ModelArrayIO command-line interface."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version

from modelarrayio.cli.export_results import _parse_export_results, export_results_main
from modelarrayio.cli.to_modelarray import _parse_to_modelarray, to_modelarray_main

COMMANDS = [
    ('to-modelarray', _parse_to_modelarray, to_modelarray_main),
    ('export-results', _parse_export_results, export_results_main),
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
