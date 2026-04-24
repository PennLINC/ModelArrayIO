from __future__ import annotations

from pathlib import Path


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
