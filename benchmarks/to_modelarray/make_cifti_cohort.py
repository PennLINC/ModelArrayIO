"""Create CIFTI dscalar benchmark cohorts.

Two modes:

``synthesize``
    Write real, seeded ``.dscalar.nii`` files (surface greyordinates at a
    configurable density) plus a long-format cohort CSV. Reproducible on any
    machine with no internal data, so the CIFTI load/runtime path is publicly
    testable.

    By default, values are drawn from scalar-appropriate marginal
    distributions (``thickness``/``area``/``sulc`` ranges) with a deterministic
    medial-wall zero block, which is realistic enough for load/runtime timing
    but still carries no real spatial autocorrelation, so it is not valid for
    compression-ratio conclusions.

    Pass ``--reference-dir`` to derive, per scalar, the shared medial-wall zero
    mask and the per-vertex mean and standard deviation across a small set of
    real ``.dscalar.nii`` files, then generate each synthetic subject as
    ``clip(mean + std * noise, observed_min, observed_max)`` with the medial
    wall zeroed. Only aggregate per-vertex statistics are read and fresh noise
    is drawn, so no real subject's row is reproduced, while the real mean map's
    spatial smoothness makes compression ratios realistic. The reference axes
    are copied verbatim so the header (and therefore ``vox_offset``) matches the
    real files exactly.

``subset``
    Take an existing long-format CIFTI cohort (for example the internal
    freesurferpost cohort) and emit a smaller deterministic slice — the first
    ``--subjects`` subjects for the selected scalars — so the real data can be
    benchmarked with the same harness without editing CSVs by hand.

Both modes emit ``subject_id,scalar_name,source_file`` rows, which
``run_benchmarks.py`` drives as a long-format (combined-output) CIFTI cohort.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import harness_common as common

# fs_LR den-164k: 163,842 vertices per hemisphere, both hemispheres, no medial
# wall removed, which is what the production freesurferpost dscalars carry.
DEFAULT_GRAYORDINATES = 327684
DEFAULT_SCALARS = ('thickness', 'area', 'sulc')

# Marginal value profiles for the no-reference synthesizer, chosen to match the
# ranges of real fs_LR surface metrics: (mean, std, min, max). These reproduce
# realistic value ranges but not real spatial autocorrelation.
SCALAR_MARGINALS: dict[str, tuple[float, float, float, float]] = {
    'thickness': (2.24, 1.02, 0.0, 5.0),
    'area': (0.52, 0.32, 0.0, 13.5),
    'sulc': (0.09, 5.35, -14.0, 17.0),
    'curv': (0.0, 0.13, -1.0, 1.0),
    'myelin': (1.6, 0.5, 0.0, 8.0),
}
# Fraction of greyordinates that fall on the medial wall (constant zero block)
# in real fs_LR den-164k dscalars.
DEFAULT_MEDIAL_WALL_FRACTION = 0.1017


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='mode', required=True)

    synth = subparsers.add_parser(
        'synthesize', help='Write seeded dscalar files and a long-format cohort CSV.'
    )
    synth.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to hold the generated .dscalar.nii files and cohort.csv.',
    )
    synth.add_argument('--subjects', type=int, default=20)
    synth.add_argument('--scalars', nargs='+', default=list(DEFAULT_SCALARS))
    synth.add_argument(
        '--grayordinates',
        type=int,
        default=DEFAULT_GRAYORDINATES,
        help='Total surface greyordinates (split evenly across two hemispheres).',
    )
    synth.add_argument('--seed', type=int, default=0)
    synth.add_argument(
        '--reference-dir',
        type=Path,
        default=None,
        help=(
            'Directory of real .dscalar.nii files to derive realistic per-vertex mean/std and '
            'the medial-wall zero mask from (matched per scalar by filename suffix). When set, '
            '--grayordinates is taken from the reference header and generated compression ratios '
            'are realistic; only aggregate statistics are read, never a real subject row.'
        ),
    )
    synth.add_argument(
        '--cohort',
        type=Path,
        help='Cohort CSV path (default: <output-dir>/cohort.csv).',
    )
    synth.set_defaults(func=synthesize)

    subset = subparsers.add_parser(
        'subset', help='Slice an existing long-format CIFTI cohort into a smaller one.'
    )
    subset.add_argument(
        '--input', type=Path, required=True, help='Existing long-format cohort CSV.'
    )
    subset.add_argument('--output', type=Path, required=True)
    subset.add_argument('--subjects', type=int, default=100)
    subset.add_argument(
        '--scalars',
        nargs='+',
        help='Scalars to keep (default: every scalar present in the input).',
    )
    subset.set_defaults(func=subset_cohort)

    return parser


def _validate_common(subjects: int) -> None:
    if subjects < 1:
        raise ValueError('--subjects must be positive')


def _build_brain_model_axis(grayordinates: int):
    """Return a two-hemisphere surface greyordinate axis of the requested length."""
    import numpy as np
    from nibabel.cifti2.cifti2_axes import BrainModelAxis

    if grayordinates < 2:
        raise ValueError('--grayordinates must be at least 2')
    left_vertices = grayordinates // 2
    right_vertices = grayordinates - left_vertices
    left = BrainModelAxis.from_surface(np.arange(left_vertices), left_vertices, name='CortexLeft')
    right = BrainModelAxis.from_surface(
        np.arange(right_vertices), right_vertices, name='CortexRight'
    )
    return left + right


def _reference_files(reference_dir: Path, scalar: str) -> list[Path]:
    """Return the real .dscalar.nii files for one scalar, matched by suffix."""
    matches = sorted(reference_dir.glob(f'**/*_{scalar}.dscalar.nii'))
    if not matches:
        raise ValueError(
            f'No reference files matching *_{scalar}.dscalar.nii under {reference_dir}.'
        )
    return matches


def _reference_profile(reference_dir: Path, scalar: str):
    """Derive the aggregate per-vertex generation profile for one scalar.

    Returns the reference brain-model axis plus per-vertex mean and standard
    deviation, the shared medial-wall zero mask, and the observed value range.
    Only these aggregates are read; individual subject rows are never emitted.
    """
    import nibabel as nb
    import numpy as np

    files = _reference_files(reference_dir, scalar)
    stack = []
    brain_axis = None
    for path in files:
        image = nb.load(str(path))
        if brain_axis is None:
            brain_axis = image.header.get_axis(1)
        stack.append(np.asarray(image.dataobj).reshape(-1).astype(np.float64))
    matrix = np.vstack(stack)
    finite = np.isfinite(matrix)
    # A vertex is medial (constant zero) when it is zero in every reference subject.
    zero_mask = np.all(matrix == 0.0, axis=0)
    with np.errstate(invalid='ignore'):
        mean = np.where(finite, matrix, np.nan)
        per_vertex_mean = np.nanmean(mean, axis=0)
        per_vertex_std = np.nan_to_num(np.nanstd(mean, axis=0), nan=0.0)
    per_vertex_mean = np.nan_to_num(per_vertex_mean, nan=0.0)
    observed = matrix[finite]
    return {
        'brain_axis': brain_axis,
        'mean': per_vertex_mean.astype(np.float32),
        'std': per_vertex_std.astype(np.float32),
        'zero_mask': zero_mask,
        'vmin': float(observed.min()),
        'vmax': float(observed.max()),
        'reference_count': len(files),
    }


def _marginal_row(scalar: str, n_elements: int, medial_mask, rng) -> object:
    """Draw one synthetic row from a scalar's marginal profile (no reference)."""
    import numpy as np

    mean, std, vmin, vmax = SCALAR_MARGINALS.get(scalar, (0.0, 1.0, -6.0, 6.0))
    values = rng.normal(mean, std, size=n_elements).astype(np.float32)
    np.clip(values, vmin, vmax, out=values)
    values[medial_mask] = 0.0
    return values.reshape(1, n_elements)


def _reference_row(profile, rng) -> object:
    """Draw one synthetic row from a reference profile; never a real row."""
    import numpy as np

    mean = profile['mean']
    std = profile['std']
    noise = rng.standard_normal(mean.shape[0]).astype(np.float32)
    values = mean + std * noise
    np.clip(values, profile['vmin'], profile['vmax'], out=values)
    values[profile['zero_mask']] = 0.0
    return values.reshape(1, mean.shape[0]).astype(np.float32)


def synthesize(args: argparse.Namespace) -> int:
    """Write the seeded cohort and return its row count."""
    import nibabel as nb
    import numpy as np
    from nibabel.cifti2.cifti2_axes import ScalarAxis

    _validate_common(args.subjects)
    if not args.scalars:
        raise ValueError('--scalars must name at least one scalar')

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.cohort or (output_dir / 'cohort.csv')

    reference_dir = getattr(args, 'reference_dir', None)
    profiles: dict[str, dict] = {}
    if reference_dir is not None:
        for scalar in args.scalars:
            profiles[scalar] = _reference_profile(reference_dir, scalar)
        n_elements = profiles[args.scalars[0]]['mean'].shape[0]
        if any(profile['mean'].shape[0] != n_elements for profile in profiles.values()):
            raise ValueError('Reference scalars have differing greyordinate counts.')
        medial_mask = None
        density = 'reference'
        summary = ', '.join(f'{s}:{profiles[s]["reference_count"]}' for s in args.scalars)
        print(f'Derived realistic profiles from {reference_dir} ({summary} files).')
    else:
        brain_axis = _build_brain_model_axis(args.grayordinates)
        n_elements = len(brain_axis)
        # A deterministic contiguous medial-wall block, matching the real zero fraction.
        medial_count = round(n_elements * DEFAULT_MEDIAL_WALL_FRACTION)
        medial_mask = np.zeros(n_elements, dtype=bool)
        medial_mask[:medial_count] = True
        density = 'den-164k' if args.grayordinates == DEFAULT_GRAYORDINATES else 'synthetic'

    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, str]] = []
    for subject_index in range(args.subjects):
        subject_id = f'sub-synth{subject_index + 1:04d}'
        subject_dir = output_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        for scalar in args.scalars:
            # The scalar becomes the CIFTI map name, whose length shifts vox_offset
            # exactly as real per-metric dscalars do, exercising the per-scalar
            # raw-read plan rather than one shared offset.
            if reference_dir is not None:
                brain_axis = profiles[scalar]['brain_axis']
                values = _reference_row(profiles[scalar], rng)
            else:
                values = _marginal_row(scalar, n_elements, medial_mask, rng)
            header = nb.cifti2.Cifti2Header.from_axes((ScalarAxis([scalar]), brain_axis))
            path = subject_dir / f'{subject_id}_space-fsLR_den-164k_{scalar}.dscalar.nii'
            nb.Cifti2Image(values, header=header).to_filename(str(path))
            rows.append(
                {
                    'subject_id': subject_id,
                    'scalar_name': scalar,
                    'source_file': str(path.resolve()),
                }
            )

    common.write_csv(cohort_path, rows, common.LONG_COHORT_FIELDS)
    print(
        f'Wrote {len(rows)} rows ({args.subjects} subjects x {len(args.scalars)} scalars, '
        f'{n_elements} greyordinates, values={density}) to {cohort_path}'
    )
    return len(rows)


def _read_long_cohort(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not {'scalar_name', 'source_file'} <= fields:
            raise ValueError(
                f'{path} is not a long-format CIFTI cohort; it must have '
                "'scalar_name' and 'source_file' columns."
            )
        return list(reader)


def _subject_key(row: dict[str, str]) -> str:
    """Return a stable subject identifier, tolerating cohorts without a session column."""
    subject = row.get('subject_id') or ''
    session = row.get('session_id') or ''
    return f'{subject}|{session}' if session else subject


def subset_cohort(args: argparse.Namespace) -> int:
    """Write the deterministic subset and return its row count."""
    _validate_common(args.subjects)
    rows = _read_long_cohort(args.input)
    if not rows:
        raise ValueError(f'{args.input} has no rows.')

    available_scalars = list(dict.fromkeys(row['scalar_name'] for row in rows))
    scalars = args.scalars or available_scalars
    missing = [scalar for scalar in scalars if scalar not in available_scalars]
    if missing:
        raise ValueError(f'Scalars not present in {args.input}: {", ".join(missing)}')

    # Subjects in first-seen order, then keep the first --subjects of them.
    ordered_subjects = list(OrderedDict.fromkeys(_subject_key(row) for row in rows))
    if len(ordered_subjects) < args.subjects:
        raise ValueError(
            f'Requested {args.subjects} subjects but {args.input} has only '
            f'{len(ordered_subjects)}.'
        )
    keep_subjects = set(ordered_subjects[: args.subjects])
    keep_scalars = set(scalars)

    selected = [
        {
            'subject_id': row.get('subject_id', ''),
            'scalar_name': row['scalar_name'],
            'source_file': row['source_file'],
        }
        for row in rows
        if row['scalar_name'] in keep_scalars and _subject_key(row) in keep_subjects
    ]
    common.write_csv(args.output, selected, common.LONG_COHORT_FIELDS)
    print(
        f'Wrote {len(selected)} rows ({args.subjects} subjects x {len(scalars)} scalars) '
        f'to {args.output}'
    )
    return len(selected)


def main() -> int:
    args = _parser().parse_args()
    args.func(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
