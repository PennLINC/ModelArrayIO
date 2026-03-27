"""Pytest hooks shared by this package's tests."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pytest

# CLI tests run subprocesses with cwd under tmp_path (outside the repo). Coverage
# discovers [tool.coverage] from the process cwd, so those children would default to
# line-only / non-parallel data and pytest-cov then fails on combine with arc data from
# the parent. Point subprocess coverage at the project config explicitly.
_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('COVERAGE_RCFILE', str(_ROOT / 'pyproject.toml'))

_FIXEL_DATA_URL = 'https://upenn.box.com/s/aaauvwrsrvsj3yvdkl1b899wx1s8ih2f.tar.gz'
_FIXEL_DATA_URL_ENV = 'MODELARRAYIO_FIXEL_TEST_DATA_URL'
_FIXEL_DATA_DIR_ENV = 'MODELARRAYIO_FIXEL_TEST_DATA_DIR'


def _download_and_extract_fixel_test_data(destination_dir: Path) -> Path:
    """Download and extract the fixel test dataset archive."""
    source_dir = os.environ.get(_FIXEL_DATA_DIR_ENV)
    if source_dir:
        source_path = Path(source_dir).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(
                f'{_FIXEL_DATA_DIR_ENV} points to a missing path: {source_path}'
            )
        return source_path

    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / 'mif_test_data.tar.gz'
    data_url = os.environ.get(_FIXEL_DATA_URL_ENV, _FIXEL_DATA_URL)
    urlretrieve(data_url, archive_path)  # noqa: S310

    with tarfile.open(archive_path, mode='r:gz') as archive:
        archive.extractall(destination_dir)  # noqa: S202

    extracted_dir = destination_dir / 'mif_test_data'
    if not extracted_dir.exists():
        raise FileNotFoundError(f'Expected extracted dataset at {extracted_dir}')
    return extracted_dir


@pytest.fixture(scope='session')
def downloaded_fixel_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide the downloaded fixel dataset directory for tests."""
    destination_dir = tmp_path_factory.mktemp('downloaded_fixel_data')
    try:
        return _download_and_extract_fixel_test_data(destination_dir)
    except (FileNotFoundError, OSError, URLError, tarfile.TarError) as exc:
        pytest.skip(f'Downloaded fixel test data unavailable: {exc}')
