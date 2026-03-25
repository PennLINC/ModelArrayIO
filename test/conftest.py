"""Pytest hooks shared by this package's tests."""

from __future__ import annotations

import os
from pathlib import Path

# CLI tests run subprocesses with cwd under tmp_path (outside the repo). Coverage
# discovers [tool.coverage] from the process cwd, so those children would default to
# line-only / non-parallel data and pytest-cov then fails on combine with arc data from
# the parent. Point subprocess coverage at the project config explicitly.
_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('COVERAGE_RCFILE', str(_ROOT / 'pyproject.toml'))
