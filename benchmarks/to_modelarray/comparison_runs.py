"""Read ``compare_revisions.py`` run directories into per-modality/backend groups.

Shared by ``create_comparison_report.py`` (the findings PDF) and ``summarize_comparison.py``
(the pinned summary CSV), so both aggregate a comparison the same way.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import harness_common as common
from output_compare import OutputReader

CORRECTNESS_KEYS = ('shape_equal', 'dtype_equal', 'sources_equal', 'nan_equal', 'values_equal')


@dataclass
class Group:
    """Aggregated original-vs-optimized medians for one modality/backend."""

    modality: str
    backend: str
    n_subjects: int
    n_elements: int
    n_scalars: int
    baseline_revision: str = ''
    baseline_wall: list[float] = field(default_factory=list)
    optimized_wall: list[float] = field(default_factory=list)
    baseline_r: list[float] = field(default_factory=list)
    optimized_r: list[float] = field(default_factory=list)
    baseline_bytes: int = 0
    optimized_bytes: int = 0
    baseline_files: int = 0
    optimized_files: int = 0
    values_equal: bool = True
    shape_equal: bool = True
    nan_equal: bool = True
    sources_equal: bool = True
    dtype_equal: bool = True
    max_abs_difference: float = 0.0

    @staticmethod
    def _median(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    @property
    def baseline_median(self) -> float:
        return self._median(self.baseline_wall)

    @property
    def optimized_median(self) -> float:
        return self._median(self.optimized_wall)

    @property
    def baseline_r_median(self) -> float:
        return self._median(self.baseline_r)

    @property
    def optimized_r_median(self) -> float:
        return self._median(self.optimized_r)

    @property
    def speedup(self) -> float:
        return self.baseline_median / self.optimized_median if self.optimized_median else 0.0

    @property
    def reduction(self) -> float:
        base = self.baseline_median
        return 100.0 * (1.0 - self.optimized_median / base) if base else 0.0

    @property
    def correctness_ok(self) -> bool:
        return all(
            (
                self.shape_equal,
                self.dtype_equal,
                self.sources_equal,
                self.nan_equal,
                self.values_equal,
            )
        )


def to_float(value: str | None) -> float | None:
    if value in (None, '', 'None'):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def output_shape(run_dir: Path, backend: str, scalar_names: list[str], modality: str) -> tuple:
    """Read (n_elements, n_subjects) from a representative optimized output."""
    scalar = scalar_names[0]
    split = modality == 'nifti'  # wide NIfTI cohorts split into one file per scalar
    output = common.output_path(
        run_dir / f'optimized-{backend}-rep-01', backend, scalar if split else None
    )
    with OutputReader(output, scalar) as reader:
        return reader.shape[1], reader.shape[0]


def load_groups(run_dirs: list[Path]) -> dict[str, list[Group]]:
    """Aggregate every comparison run directory into per-modality groups."""
    groups: dict[tuple[str, str], Group] = {}
    for run_dir in run_dirs:
        meta = json.loads((run_dir / 'comparison_meta.json').read_text(encoding='utf-8'))
        modality = meta['modality']
        scalar_names = meta.get('scalar_names') or meta.get('scalar_columns') or []
        n_scalars = len(scalar_names)
        with (run_dir / 'comparison_results.csv').open(newline='', encoding='utf-8') as stream:
            comparison_rows = list(csv.DictReader(stream))
        correctness_path = run_dir / 'correctness_results.csv'
        correctness_rows = []
        if correctness_path.exists():
            with correctness_path.open(newline='', encoding='utf-8') as stream:
                correctness_rows = list(csv.DictReader(stream))
        backends = sorted({row['backend'] for row in comparison_rows})
        for backend in backends:
            n_elements, n_subjects = output_shape(run_dir, backend, scalar_names, modality)
            group = Group(
                modality=modality,
                backend=backend,
                n_subjects=n_subjects,
                n_elements=n_elements,
                n_scalars=n_scalars,
                baseline_revision=meta.get('baseline_revision') or '',
            )
            for row in comparison_rows:
                if row['backend'] != backend:
                    continue
                wall = to_float(row.get('wall_seconds'))
                r_read = to_float(row.get('r_read_seconds'))
                if row['implementation'] == 'baseline':
                    group.baseline_wall.append(wall)
                    group.baseline_r.append(r_read)
                    group.baseline_bytes = int(row.get('output_bytes') or 0)
                    group.baseline_files = int(row.get('output_file_count') or 0)
                else:
                    group.optimized_wall.append(wall)
                    group.optimized_r.append(r_read)
                    group.optimized_bytes = int(row.get('output_bytes') or 0)
                    group.optimized_files = int(row.get('output_file_count') or 0)
            backend_correctness = [row for row in correctness_rows if row['backend'] == backend]
            for row in backend_correctness:
                for key in CORRECTNESS_KEYS:
                    setattr(group, key, getattr(group, key) and as_bool(row.get(key)))
                group.max_abs_difference = max(
                    group.max_abs_difference, to_float(row.get('max_abs_difference')) or 0.0
                )
            groups[(modality, backend)] = group
    grouped: dict[str, list[Group]] = {}
    for group in groups.values():
        grouped.setdefault(group.modality, []).append(group)
    for modality_groups in grouped.values():
        modality_groups.sort(key=lambda g: g.backend)
    return grouped
