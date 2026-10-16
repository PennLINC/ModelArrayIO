"""Aggregate ``compare_revisions.py`` run directories into one pinned summary CSV.

Reads the per-modality original-vs-optimized run directories and writes a single
committed summary row per modality/backend, so a comparison remains auditable and
regressions are detectable by rerunning one command. The full findings PDF is
rendered separately by ``create_comparison_report.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import harness_common as common
from comparison_runs import Group, load_groups


def _summary_row(group: Group) -> dict[str, object]:
    return {
        'modality': group.modality,
        'backend': group.backend,
        'n_subjects': group.n_subjects,
        'n_elements': group.n_elements,
        'n_scalars': group.n_scalars,
        'baseline_revision': group.baseline_revision,
        'baseline_median_wall_seconds': round(group.baseline_median, 4),
        'optimized_median_wall_seconds': round(group.optimized_median, 4),
        'speedup': round(group.speedup, 4),
        'wall_time_reduction_percent': round(group.reduction, 2),
        'baseline_median_r_read_seconds': round(group.baseline_r_median, 4),
        'optimized_median_r_read_seconds': round(group.optimized_r_median, 4),
        'baseline_output_bytes': group.baseline_bytes,
        'optimized_output_bytes': group.optimized_bytes,
        'baseline_output_file_count': group.baseline_files,
        'optimized_output_file_count': group.optimized_files,
        'values_equal': group.correctness_ok,
        'max_abs_difference': group.max_abs_difference,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True, dest='run_dirs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    grouped = load_groups([run_dir.expanduser().resolve() for run_dir in args.run_dirs])
    rows = [_summary_row(group) for groups in grouped.values() for group in groups]
    common.write_csv(args.output, rows)
    print(f'Wrote {len(rows)} rows to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
