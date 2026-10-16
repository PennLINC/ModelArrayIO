"""Identify the exact source tree a benchmark ran against, committed or not.

``git rev-parse HEAD`` is misleading for an uncommitted working tree (and for a
tree copied to a cluster with rsync): it names the last commit, not the code
that ran. This module hashes the content of every tracked-or-untracked,
non-ignored file under the paths that affect conversion results, so two hosts
holding the same working tree report the same ``source_tree_sha256`` and any
drift (a forgotten rsync, a local edit) is visible in the recorded metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

IDENTITY_PATHS = ('src', 'pyproject.toml', 'benchmarks')
_SKIP_DIRS = {'.venv', '__pycache__', 'data', 'results', 'cohorts', '.git'}


def _git(repo: Path, *args: str) -> str | None:
    executable = shutil.which('git')
    if executable is None or not (repo / '.git').exists():
        return None
    completed = subprocess.run(
        [executable, *args], cwd=repo, capture_output=True, text=True, check=False
    )
    return completed.stdout if completed.returncode == 0 else None


def _listed_files(repo: Path) -> list[Path]:
    """Return tracked + untracked (not ignored) files under the identity paths."""
    listing = _git(repo, 'ls-files', '-co', '--exclude-standard', '--', *IDENTITY_PATHS)
    if listing is not None:
        return sorted(
            repo / line for line in listing.splitlines() if line and (repo / line).is_file()
        )
    files: list[Path] = []
    for root_name in IDENTITY_PATHS:
        root = repo / root_name
        if root.is_file():
            files.append(root)
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            files.extend(Path(directory) / name for name in sorted(filenames))
    return sorted(files)


def source_tree_sha256(repo: Path) -> str:
    """Hash sorted (relative path, bytes) pairs of every identity file."""
    digest = hashlib.sha256()
    for path in _listed_files(repo):
        digest.update(path.relative_to(repo).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def source_identity(repo: str | Path) -> dict[str, object]:
    """Return ``commit``, ``dirty`` and ``source_tree_sha256`` for a repository."""
    root = Path(repo).resolve()
    commit = _git(root, 'rev-parse', 'HEAD')
    status = _git(root, 'status', '--porcelain', '--', *IDENTITY_PATHS)
    return {
        'commit': commit.strip() if commit else None,
        'dirty': bool(status and status.strip()),
        'source_tree_sha256': source_tree_sha256(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(source_identity(args.repo), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
