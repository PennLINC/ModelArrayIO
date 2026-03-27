"""Data helpers for HDF5 benchmark tests using real S3 voxel templates.

The benchmark dataset is built from open ABIDE ALFF/mask files on S3 (same
workflow pattern as ``test/test_voxels_s3.py``), then expanded with controlled
random variation to emulate larger cohorts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from modelarrayio.utils.s3_utils import load_nibabel
from modelarrayio.utils.voxels import flattened_image

logger = logging.getLogger(__name__)

_U64_FLOAT_DENOM = float(2**64)
_SALT_DROPOUT = np.uint64(0xA24BAED4963EE407)
_SALT_NOISE_1 = np.uint64(0x9FB21C651E98DF25)
_SALT_NOISE_2 = np.uint64(0xC13FA9A902A6328F)
_SUBJECT_MUL = np.uint64(0x9E3779B185EBCA87)
_VOXEL_MUL = np.uint64(0xC2B2AE3D27D4EB4F)

# Open ABIDE OHSU subjects used in test/test_voxels_s3.py
_OHSU_SUBJECTS = [
    'OHSU_0050142',
    'OHSU_0050143',
    'OHSU_0050144',
    'OHSU_0050145',
]
_BUCKET = 'fcp-indi'
_PREFIX = 'data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global'


@dataclass(frozen=True)
class SyntheticBenchmarkDataset:
    """Container for benchmark-ready row vectors and generation metadata."""

    rows: list[np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True)
class SyntheticBenchmarkPlan:
    """Precomputed benchmark synthesis plan for streamed stripe generation."""

    sampled_templates: np.ndarray
    voxel_mean: np.ndarray
    voxel_sd: np.ndarray
    template_indices: np.ndarray
    subject_scales: np.ndarray
    subject_offsets: np.ndarray
    dropout_probs: np.ndarray
    subject_key_base_u64: np.ndarray
    dropout_thresholds_u64: np.ndarray
    noise_std: float
    seed: int
    metadata: dict[str, object]

    @property
    def num_subjects(self) -> int:
        return int(self.template_indices.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.voxel_mean.shape[0])


def _s3_alff(subject_id: str) -> str:
    return f's3://{_BUCKET}/{_PREFIX}/alff/{subject_id}_alff.nii.gz'


def _s3_mask(subject_id: str) -> str:
    return f's3://{_BUCKET}/{_PREFIX}/func_mask/{subject_id}_func_mask.nii.gz'


@lru_cache(maxsize=2)
def _load_s3_template_rows(
    allow_fallback: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Load and cache real ALFF template rows from S3."""
    os.environ.setdefault('MODELARRAYIO_S3_ANON', '1')
    try:
        group_mask_img = load_nibabel(_s3_mask(_OHSU_SUBJECTS[0]))
        group_mask_matrix = group_mask_img.get_fdata() > 0
        volume_shape = tuple(int(v) for v in group_mask_img.shape[:3])

        template_rows: list[np.ndarray] = []
        for subject in _OHSU_SUBJECTS:
            scalar_img = load_nibabel(_s3_alff(subject))
            subject_mask_img = load_nibabel(_s3_mask(subject))
            row = flattened_image(scalar_img, subject_mask_img, group_mask_matrix).astype(
                np.float32
            )
            template_rows.append(row)

        if not template_rows:
            raise ValueError('failed to load S3 template rows')
        return group_mask_matrix, np.vstack(template_rows), volume_shape
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if not allow_fallback:
            raise RuntimeError(
                'Failed to load required S3 benchmark templates. '
                'Install modelarrayio[s3], enable network access, and set MODELARRAYIO_S3_ANON=1.'
            ) from exc
        # Keep benchmark tests runnable in environments without boto3/network.
        logger.warning('Falling back to local synthetic templates for benchmarks: %s', exc)
        return _load_local_fallback_template_rows()


def _load_local_fallback_template_rows() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Build deterministic local template rows when S3 templates are unavailable."""
    rng = np.random.default_rng(20260313)
    volume_shape = (16, 16, 8)
    group_mask_matrix = rng.random(volume_shape) > 0.25
    num_voxels = int(group_mask_matrix.sum())

    base_signal = rng.normal(loc=0.0, scale=1.0, size=num_voxels).astype(np.float32)
    template_rows: list[np.ndarray] = []
    for _ in _OHSU_SUBJECTS:
        scale = float(rng.normal(loc=1.0, scale=0.15))
        offset = float(rng.normal(loc=0.0, scale=0.2))
        row = (base_signal * scale + offset).astype(np.float32)
        dropout = rng.random(num_voxels) < 0.05
        row[dropout] = np.nan
        template_rows.append(row)
    return group_mask_matrix, np.vstack(template_rows), volume_shape


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 mixing for deterministic pseudo-random u64 values."""
    z = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64, copy=False)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _u64_to_unit_float(values: np.ndarray) -> np.ndarray:
    # Map uint64 to (0, 1) deterministically.
    return (values.astype(np.float64) + 0.5) / _U64_FLOAT_DENOM


@lru_cache(maxsize=32)
def make_realistic_voxel_benchmark_plan(
    num_input_files: int,
    *,
    seed: int = 8675309,
    max_voxels: int = 0,
    noise_std: float = 0.35,
    dropout_range: tuple[float, float] = (0.01, 0.08),
    require_s3_templates: bool = False,
) -> SyntheticBenchmarkPlan:
    """Build a deterministic benchmark synthesis plan for streamed writes.

    This avoids materializing the full (subjects x items) matrix in memory.
    """
    if num_input_files <= 0:
        raise ValueError('num_input_files must be positive')
    if max_voxels < 0:
        max_voxels = 0

    min_dropout, max_dropout = dropout_range
    if not (0.0 <= min_dropout <= max_dropout < 1.0):
        raise ValueError('dropout_range must satisfy 0 <= min <= max < 1')

    rng = np.random.default_rng(seed)
    group_mask_matrix, template_rows, volume_shape = _load_s3_template_rows(
        allow_fallback=not require_s3_templates
    )
    num_template_rows, total_items = template_rows.shape
    if total_items == 0:
        raise ValueError('S3 template rows are empty')

    if max_voxels == 0 or max_voxels >= total_items:
        selected_indices = np.arange(total_items, dtype=np.int64)
    else:
        selected_indices = np.sort(rng.choice(total_items, size=max_voxels, replace=False))
    sampled_templates = template_rows[:, selected_indices].astype(np.float32, copy=False)

    voxel_mean = np.nanmean(sampled_templates, axis=0).astype(np.float32)
    voxel_sd = np.nanstd(sampled_templates, axis=0).astype(np.float32)
    finite_sd = np.isfinite(voxel_sd) & (voxel_sd > 1e-6)
    sd_floor = float(np.median(voxel_sd[finite_sd])) if np.any(finite_sd) else 1.0
    voxel_sd = np.where(finite_sd, voxel_sd, sd_floor).astype(np.float32)

    finite_mean = np.isfinite(voxel_mean)
    mean_fill = float(np.mean(voxel_mean[finite_mean])) if np.any(finite_mean) else 0.0
    voxel_mean = np.where(finite_mean, voxel_mean, mean_fill).astype(np.float32)

    template_indices = rng.integers(0, num_template_rows, size=num_input_files, dtype=np.int64)
    subject_scales = rng.normal(loc=1.0, scale=0.12, size=num_input_files).astype(np.float32)
    subject_offsets = rng.normal(loc=0.0, scale=0.25, size=num_input_files).astype(np.float32)
    dropout_probs = rng.uniform(min_dropout, max_dropout, size=num_input_files).astype(np.float32)
    seed_u64 = np.uint64(seed)
    subject_key_base_u64 = (
        np.arange(num_input_files, dtype=np.uint64)[:, np.newaxis] * _SUBJECT_MUL
    ) ^ seed_u64
    dropout_thresholds_u64 = (dropout_probs.astype(np.float64) * _U64_FLOAT_DENOM).astype(
        np.uint64
    )[:, np.newaxis]

    uses_s3_templates = template_rows.shape[1] > 0 and volume_shape != (16, 16, 8)
    metadata: dict[str, object] = {
        'seed': seed,
        'num_input_files': num_input_files,
        'volume_shape': list(volume_shape),
        'group_mask_voxels': int(group_mask_matrix.sum()),
        'sampled_voxels': int(selected_indices.shape[0]),
        'noise_std': float(noise_std),
        'adaptive_sd_median': float(np.median(voxel_sd)),
        'adaptive_sd_mean': float(np.mean(voxel_sd)),
        'dropout_range': [float(min_dropout), float(max_dropout)],
        # Filled after streamed generation/writing.
        'mean_missing_fraction': float('nan'),
        'std_missing_fraction': float('nan'),
        'template_subjects': _OHSU_SUBJECTS if uses_s3_templates else ['local_synthetic_template'],
        'workflow_reference': (
            'Open S3 ABIDE ALFF + func-mask templates with per-voxel mean/SD adaptive variation'
            if uses_s3_templates
            else 'Local synthetic fallback templates with per-voxel mean/SD adaptive variation'
        ),
    }
    return SyntheticBenchmarkPlan(
        sampled_templates=sampled_templates,
        voxel_mean=voxel_mean,
        voxel_sd=voxel_sd,
        template_indices=template_indices,
        subject_scales=subject_scales,
        subject_offsets=subject_offsets,
        dropout_probs=dropout_probs,
        subject_key_base_u64=subject_key_base_u64,
        dropout_thresholds_u64=dropout_thresholds_u64,
        noise_std=float(noise_std),
        seed=int(seed),
        metadata=metadata,
    )


def fill_realistic_voxel_stripe(
    plan: SyntheticBenchmarkPlan,
    *,
    start: int,
    end: int,
    out: np.ndarray,
) -> np.ndarray:
    """Fill ``out`` with synthetic values for a column stripe and return NaN counts.

    Parameters
    ----------
    plan
        Precomputed synthesis plan.
    start, end
        Stripe bounds in [0, num_items].
    out
        Destination array with shape (num_subjects, >= stripe_width), dtype float32.
    """
    if start < 0 or end <= start or end > plan.num_items:
        raise ValueError(
            f'invalid stripe bounds: start={start}, end={end}, num_items={plan.num_items}'
        )
    stripe_width = end - start
    if out.shape[0] != plan.num_subjects or out.shape[1] < stripe_width:
        raise ValueError(
            f'out shape {out.shape} incompatible with plan (subjects={plan.num_subjects}, '
            f'stripe_width={stripe_width})'
        )

    view = out[:, :stripe_width]
    mean_slice = plan.voxel_mean[start:end]
    sd_slice = plan.voxel_sd[start:end]
    template_slice = plan.sampled_templates[plan.template_indices, start:end]

    view[:, :] = (
        mean_slice[np.newaxis, :]
        + plan.subject_scales[:, np.newaxis] * (template_slice - mean_slice[np.newaxis, :])
        + plan.subject_offsets[:, np.newaxis] * sd_slice[np.newaxis, :]
    )

    # Stateless RNG keyed by (seed, subject, voxel) keeps values invariant
    # across stripe/chunk geometry.
    voxel_ids = np.arange(start, end, dtype=np.uint64)[np.newaxis, :]
    key_matrix = plan.subject_key_base_u64 ^ (voxel_ids * _VOXEL_MUL)

    dropout_hash = _splitmix64(key_matrix ^ _SALT_DROPOUT)
    drop_mask = dropout_hash < plan.dropout_thresholds_u64
    view[drop_mask] = np.nan

    valid_mask = np.isfinite(view)
    if np.any(valid_mask):
        u1 = _u64_to_unit_float(_splitmix64(key_matrix ^ _SALT_NOISE_1))
        u2 = _u64_to_unit_float(_splitmix64(key_matrix ^ _SALT_NOISE_2))
        u1 = np.clip(u1, 1e-12, 1.0)
        noise = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        scaled_noise = (
            noise.astype(np.float32, copy=False) * (plan.noise_std * sd_slice)[np.newaxis, :]
        )
        view[valid_mask] = view[valid_mask] + scaled_noise[valid_mask]

    return np.count_nonzero(np.isnan(view), axis=1)


def finalize_plan_metadata(
    plan: SyntheticBenchmarkPlan,
    *,
    mean_missing_fraction: float,
    std_missing_fraction: float,
) -> dict[str, object]:
    metadata = dict(plan.metadata)
    metadata['mean_missing_fraction'] = float(mean_missing_fraction)
    metadata['std_missing_fraction'] = float(std_missing_fraction)
    return metadata


def make_realistic_voxel_benchmark_dataset(
    num_input_files: int,
    *,
    seed: int = 8675309,
    max_voxels: int = 0,
    noise_std: float = 0.35,
    dropout_range: tuple[float, float] = (0.01, 0.08),
    require_s3_templates: bool = False,
) -> SyntheticBenchmarkDataset:
    """Generate benchmark rows from real S3 ALFF templates plus random variation.

    Parameters
    ----------
    num_input_files
        Number of synthetic subjects / input files to emulate.
    seed
        Seed for deterministic generation.
    max_voxels
        Max number of group-mask voxels sampled into the benchmark matrix.
        ``0`` uses the full group mask.
    noise_std
        Additive noise scale in units of per-voxel SD estimated from S3 templates.
    dropout_range
        Inclusive range for per-subject mask dropout probability.
    """
    if num_input_files <= 0:
        raise ValueError('num_input_files must be positive')
    if max_voxels < 0:
        max_voxels = 0

    plan = make_realistic_voxel_benchmark_plan(
        num_input_files=num_input_files,
        seed=seed,
        max_voxels=max_voxels,
        noise_std=noise_std,
        dropout_range=dropout_range,
        require_s3_templates=require_s3_templates,
    )
    rows: list[np.ndarray] = []
    for subject_idx in range(plan.num_subjects):
        row = (
            plan.voxel_mean
            + plan.subject_scales[subject_idx]
            * (plan.sampled_templates[plan.template_indices[subject_idx]] - plan.voxel_mean)
            + plan.subject_offsets[subject_idx] * plan.voxel_sd
        ).astype(np.float32, copy=False)

        row_seed = int(
            (np.uint64(plan.seed) << np.uint64(1)) ^ np.uint64(subject_idx + 0xA5A5A5A5)
        )
        row_rng = np.random.default_rng(row_seed)
        row[row_rng.random(plan.num_items, dtype=np.float32) < plan.dropout_probs[subject_idx]] = (
            np.nan
        )

        valid = np.isfinite(row)
        if np.any(valid):
            noise = row_rng.normal(loc=0.0, scale=1.0, size=plan.num_items).astype(
                np.float32, copy=False
            )
            row[valid] = row[valid] + noise[valid] * (plan.noise_std * plan.voxel_sd[valid])
        rows.append(row)

    missing_fractions = np.array([float(np.isnan(row).mean()) for row in rows], dtype=np.float32)
    metadata = finalize_plan_metadata(
        plan,
        mean_missing_fraction=float(np.mean(missing_fractions)),
        std_missing_fraction=float(np.std(missing_fractions)),
    )
    return SyntheticBenchmarkDataset(rows=rows, metadata=metadata)
