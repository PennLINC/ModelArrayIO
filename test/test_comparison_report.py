"""Tests for the original-vs-optimized findings report (``create_comparison_report.py``)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# The report renders PDFs with ReportLab, an optional benchmark tool that is not a package
# dependency; skip these tests (rather than fail the whole session) when it is not installed.
pytest.importorskip('reportlab')

import comparison_runs
import create_comparison_report as comparison_report

# --- comparison report: production-scale section ------------------------------


def test_scale_section_renders_every_step_and_is_empty_without_rows(tmp_path: Path) -> None:
    styles = comparison_report._styles()
    assert comparison_report._scale_section([], 'note', styles) == []

    scale = tmp_path / 'scale.csv'
    with scale.open('w', newline='') as stream:
        fields = [
            'step',
            'scalars',
            'subjects',
            'backend',
            'workers',
            'baseline_median_wall_seconds',
            'optimized_median_wall_seconds',
            'speedup',
            'input_files_per_minute',
            'peak_memory_gb',
            'fragment_count',
            'fast_path_rows',
            'passed',
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                'step': '3_full_scalar_original_vs_optimized',
                'scalars': 'area',
                'subjects': '8693',
                'backend': 'hdf5',
                'workers': '1',
                'baseline_median_wall_seconds': '11854.8',
                'optimized_median_wall_seconds': '429.9',
                'speedup': '27.57',
            }
        )
        writer.writerow(
            {
                'step': '3_full_scalar_worker_sweep',
                'scalars': 'area',
                'subjects': '8693',
                'backend': 'tiledb',
                'workers': '8',
                'optimized_median_wall_seconds': '68.6',
                'input_files_per_minute': '7603',
                'peak_memory_gb': '20.4',
                'fragment_count': '1',
                'fast_path_rows': '8692',
            }
        )
        writer.writerow(
            {
                'step': '4_scalar_sharded_array',
                'scalars': 'curv',
                'subjects': '8693',
                'backend': 'hdf5',
                'workers': '8',
                'optimized_median_wall_seconds': '134.5',
                'passed': 'True',
            }
        )
    rows = comparison_report._load_scale(scale)
    assert len(rows) == 3
    flowables = comparison_report._scale_section(rows, 'Original run: 112 h first scalar.', styles)
    # A page break, heading, note, and one heading+table per populated step.
    assert len(flowables) >= 8
    assert comparison_report._load_scale(None) == []


# --- comparison report: text follows the measured runs ----------------------------------


def _group(modality: str, backend: str, baseline: float, optimized: float, *, equal: bool = True):
    return comparison_runs.Group(
        modality=modality,
        backend=backend,
        n_subjects=100,
        n_elements=10,
        n_scalars=1,
        baseline_wall=[baseline],
        optimized_wall=[optimized],
        values_equal=equal,
    )


def _story_text(flowables) -> str:
    """Flatten the report story, including paragraphs nested in table cells."""
    parts = []
    for flowable in flowables:
        if hasattr(flowable, 'getPlainText'):
            parts.append(flowable.getPlainText())
        parts.extend(
            _story_text(cell for cell in row if not isinstance(cell, str))
            for row in getattr(flowable, '_cellvalues', [])
        )
    return ' '.join(parts)


def _report_meta(grouped) -> dict[str, object]:
    return {
        'baseline_revision': '9ab13dd',
        'platform': 'CUBIC compute nodes',
        'versions': dict.fromkeys(('python', 'numpy', 'nibabel', 'h5py', 'tiledb'), '1.0'),
        'settings': {'repetitions': '3', 'warmups': '1', 'workers': '1', 'target_chunk_mb': '2'},
        'cohorts': {
            modality: {
                'cohort': f'/c/{modality}.csv',
                'cohort_sha256': 'ab' * 32,
                'scalar_names': ['x'],
            }
            for modality in grouped
        },
        'cohort_sources': {'cifti': 'RBC subset, internal & real'},
    }


def test_report_text_names_what_was_measured() -> None:
    # The pinned CUBIC medians: TileDB beats HDF5 for NIfTI and MIF, so no HDF5 claim is made.
    cubic = [
        _group('nifti', 'hdf5', 3.02, 2.17),
        _group('nifti', 'tiledb', 3.41, 2.33),
        _group('cifti', 'hdf5', 385.78, 12.12),
        _group('cifti', 'tiledb', 396.47, 12.56),
        _group('mif', 'hdf5', 10.76, 2.91),
        _group('mif', 'tiledb', 14.52, 3.53),
    ]
    assert comparison_report._scope(cubic) == 'NIfTI, CIFTI, and MIF on both HDF5 and TileDB'
    assert comparison_report._largest_gain(cubic) == (
        'The largest gain is CIFTI on HDF5 (31.8x), where the raw-read fast path removes the '
        'per-file XML header parse.'
    )
    mif_only = [_group('mif', 'hdf5', 2.0, 1.0), _group('mif', 'tiledb', 4.0, 1.0)]
    assert comparison_report._scope(mif_only) == 'MIF on both HDF5 and TileDB'
    assert 'one dense fragment' in comparison_report._largest_gain(mif_only)
    assert comparison_report._scope([_group('nifti', 'hdf5', 2.0, 1.0)]) == 'NIfTI on HDF5'

    assert comparison_report._count('3', 'repetition') == 'three repetitions'
    assert comparison_report._count('1', 'warm-up') == 'one warm-up'
    assert comparison_report._count('0', 'warm-up') == 'no warm-ups'
    assert comparison_report._count('1/3', 'repetition') == '1/3 repetitions'
    settings = [{'target_chunk_mb': 32}, {'target_chunk_mb': 2.0}, {}]
    assert comparison_report._run_setting(settings, 'target_chunk_mb') == '2/32'
    with pytest.raises(SystemExit, match='MODALITY=TEXT'):
        comparison_report._cohort_sources(['dwi=somewhere'])


def test_report_verdict_follows_the_results() -> None:
    styles = comparison_report._styles()
    passing = {'cifti': [_group('cifti', 'hdf5', 30.0, 3.0), _group('cifti', 'tiledb', 30.0, 3.0)]}
    text = _story_text(comparison_report._report_story(passing, _report_meta(passing), styles))
    assert 'It is safe to adopt' in text
    assert 'for CIFTI on both HDF5 and TileDB' in text
    assert 'three counterbalanced repetitions after one discarded warm-up' in text
    assert 'RBC subset, internal & real' in text
    assert 'synth' not in text
    assert 'Local' not in text

    failing = {
        'nifti': [
            _group('nifti', 'hdf5', 1.0, 2.0),
            _group('nifti', 'tiledb', 2.0, 1.0, equal=False),
        ]
    }
    text = _story_text(comparison_report._report_story(failing, _report_meta(failing), styles))
    assert 'safe to adopt' not in text
    assert 'in 1 of the 2 modality x backend configurations' in text
    assert 'but only 1 of 2 optimized outputs were bitwise equal' in text
    assert 'faster in 1 of 2 configurations and bitwise equal in 1 of 2' in text
    assert '1 of 2 configurations passed every check' in text
    assert '100 subjects x 1 scalar:' in text
