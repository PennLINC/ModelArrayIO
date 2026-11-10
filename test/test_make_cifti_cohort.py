"""Tests for the CIFTI benchmark cohort generator."""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

import harness_common
import make_cifti_cohort as generator

_ONE_THICKNESS_ROW = {
    'subject_id': 'sub-a',
    'scalar_name': 'thickness',
    'source_file': '/data/a.dscalar.nii',
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def test_synthesize_writes_real_dscalars_and_long_cohort(tmp_path) -> None:
    args = generator._parser().parse_args(
        [
            'synthesize',
            '--output-dir',
            str(tmp_path / 'data'),
            '--subjects',
            '3',
            '--scalars',
            'thickness',
            'area',
            '--grayordinates',
            '400',
            '--seed',
            '7',
        ]
    )
    row_count = args.func(args)
    assert row_count == 6  # 3 subjects x 2 scalars

    rows = _read(tmp_path / 'data' / 'cohort.csv')
    assert list(rows[0].keys()) == ['subject_id', 'scalar_name', 'source_file']
    assert {row['scalar_name'] for row in rows} == {'thickness', 'area'}

    # Every source is a real dscalar with the requested greyordinate count.
    for row in rows:
        image = nb.load(row['source_file'])
        assert isinstance(image, nb.Cifti2Image)
        assert image.shape == (1, 400)


def test_synthesize_is_deterministic_for_a_seed(tmp_path) -> None:
    def _run(target: Path) -> np.ndarray:
        args = generator._parser().parse_args(
            [
                'synthesize',
                '--output-dir',
                str(target),
                '--subjects',
                '1',
                '--scalars',
                'thickness',
                '--grayordinates',
                '256',
                '--seed',
                '11',
            ]
        )
        args.func(args)
        source = _read(target / 'cohort.csv')[0]['source_file']
        return nb.load(source).get_fdata(dtype=np.float32).reshape(-1)

    np.testing.assert_array_equal(_run(tmp_path / 'a'), _run(tmp_path / 'b'))


def test_synthesize_varies_offset_per_scalar(tmp_path) -> None:
    # Different map-name lengths must shift vox_offset, matching real per-metric
    # dscalars and exercising the per-scalar raw-read plan.
    args = generator._parser().parse_args(
        [
            'synthesize',
            '--output-dir',
            str(tmp_path / 'data'),
            '--subjects',
            '1',
            '--scalars',
            'curv',
            'desc-fwhm25_thickness',
            '--grayordinates',
            '256',
        ]
    )
    args.func(args)
    rows = _read(tmp_path / 'data' / 'cohort.csv')
    offsets = {row['scalar_name']: nb.load(row['source_file']).dataobj.offset for row in rows}
    assert offsets['curv'] != offsets['desc-fwhm25_thickness']


def test_synthesize_default_has_medial_wall_zeros_and_realistic_range(tmp_path) -> None:
    # The no-reference default must carry a constant zero block (medial wall)
    # and scalar-appropriate value ranges rather than raw standard-normal noise.
    args = generator._parser().parse_args(
        [
            'synthesize',
            '--output-dir',
            str(tmp_path / 'data'),
            '--subjects',
            '2',
            '--scalars',
            'thickness',
            '--grayordinates',
            '1000',
            '--seed',
            '3',
        ]
    )
    args.func(args)
    rows = _read(tmp_path / 'data' / 'cohort.csv')
    values = nb.load(rows[0]['source_file']).get_fdata(dtype=np.float32).reshape(-1)
    assert float(values.min()) >= 0.0  # thickness is non-negative
    assert float(values.max()) <= 5.0
    assert np.mean(values == 0.0) > 0.05  # a real medial-wall-sized zero block


def _write_reference_dscalar(path: Path, values: np.ndarray, scalar: str) -> None:
    from nibabel.cifti2.cifti2_axes import ScalarAxis

    n_elements = values.shape[-1]
    brain_axis = generator._build_brain_model_axis(n_elements)
    header = nb.cifti2.Cifti2Header.from_axes((ScalarAxis([scalar]), brain_axis))
    path.parent.mkdir(parents=True, exist_ok=True)
    nb.Cifti2Image(values.reshape(1, n_elements).astype(np.float32), header=header).to_filename(
        str(path)
    )


def test_synthesize_reference_mode_matches_stats_without_copying_rows(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n_elements = 800
    zero_mask = np.zeros(n_elements, dtype=bool)
    zero_mask[:100] = True  # a known medial-wall block shared across subjects
    reference_dir = tmp_path / 'reference'
    reference_rows = []
    for index in range(3):
        row = rng.normal(2.2, 1.0, size=n_elements).astype(np.float32)
        np.clip(row, 0.0, 5.0, out=row)
        row[zero_mask] = 0.0
        reference_rows.append(row)
        _write_reference_dscalar(
            reference_dir / f'sub-real{index}' / f'sub-real{index}_thickness.dscalar.nii',
            row,
            'thickness',
        )

    args = generator._parser().parse_args(
        [
            'synthesize',
            '--output-dir',
            str(tmp_path / 'out'),
            '--subjects',
            '5',
            '--scalars',
            'thickness',
            '--reference-dir',
            str(reference_dir),
            '--seed',
            '1',
        ]
    )
    args.func(args)

    rows = _read(tmp_path / 'out' / 'cohort.csv')
    assert len(rows) == 5
    generated = np.vstack(
        [nb.load(row['source_file']).get_fdata(dtype=np.float32).reshape(-1) for row in rows]
    )
    # The medial-wall block is preserved exactly.
    assert np.all(generated[:, zero_mask] == 0.0)
    # Values stay within the observed reference range.
    assert float(generated.min()) >= 0.0
    assert float(generated.max()) <= 5.0
    # Fresh noise is drawn: no generated row equals any reference row.
    reference = np.vstack(reference_rows)
    for generated_row in generated:
        assert not any(np.array_equal(generated_row, real_row) for real_row in reference)


def test_subset_selects_first_subjects_and_requested_scalars(tmp_path) -> None:
    source = tmp_path / 'full.csv'
    harness_common.write_csv(
        source,
        [
            {
                'subject_id': subject,
                'session_id': 'ses-1',
                'scalar_name': scalar,
                'source_file': f'/data/{subject}_{scalar}.dscalar.nii',
            }
            for subject in ('sub-a', 'sub-b', 'sub-c')
            for scalar in ('thickness', 'area', 'sulc')
        ],
    )

    output = tmp_path / 'subset.csv'
    args = generator._parser().parse_args(
        [
            'subset',
            '--input',
            str(source),
            '--output',
            str(output),
            '--subjects',
            '2',
            '--scalars',
            'thickness',
            'sulc',
        ]
    )
    row_count = args.func(args)

    rows = _read(output)
    assert row_count == 4  # 2 subjects x 2 scalars
    assert {row['subject_id'] for row in rows} == {'sub-a', 'sub-b'}
    assert {row['scalar_name'] for row in rows} == {'thickness', 'sulc'}


def test_subset_rejects_more_subjects_than_available(tmp_path) -> None:
    source = tmp_path / 'full.csv'
    harness_common.write_csv(source, [_ONE_THICKNESS_ROW])

    args = generator._parser().parse_args(
        [
            'subset',
            '--input',
            str(source),
            '--output',
            str(tmp_path / 'out.csv'),
            '--subjects',
            '5',
        ]
    )
    with pytest.raises(ValueError, match='only 1'):
        args.func(args)


def test_subset_rejects_unknown_scalar(tmp_path) -> None:
    source = tmp_path / 'full.csv'
    harness_common.write_csv(source, [_ONE_THICKNESS_ROW])

    args = generator._parser().parse_args(
        [
            'subset',
            '--input',
            str(source),
            '--output',
            str(tmp_path / 'out.csv'),
            '--subjects',
            '1',
            '--scalars',
            'nope',
        ]
    )
    with pytest.raises(ValueError, match='not present'):
        args.func(args)


def test_subset_rejects_wide_format_cohort(tmp_path) -> None:
    source = tmp_path / 'wide.csv'
    harness_common.write_csv(
        source, [{'subject_id': 'sub-a', 'alff': 's3://bucket/a_alff.nii.gz'}]
    )

    args = generator._parser().parse_args(
        [
            'subset',
            '--input',
            str(source),
            '--output',
            str(tmp_path / 'out.csv'),
            '--subjects',
            '1',
        ]
    )
    with pytest.raises(ValueError, match='long-format'):
        args.func(args)
