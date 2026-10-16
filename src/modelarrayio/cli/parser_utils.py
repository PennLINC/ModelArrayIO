from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from modelarrayio.utils.s3_utils import is_s3_path


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


def _is_file_or_s3(path: str | Path | None, parser) -> Path | str:
    """Accept an existing local file or a syntactically valid S3 object URI."""
    if path is not None and is_s3_path(path):
        parsed = urlparse(str(path))
        if parsed.netloc and parsed.path.lstrip('/'):
            return str(path)
        raise parser.error(f'Invalid S3 URI: <{path}>.')
    return _is_file(path, parser)
