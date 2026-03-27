"""Diagnostic image helpers for conversion commands."""

from __future__ import annotations

from pathlib import Path

import nibabel as nb
import numpy as np

from modelarrayio.utils.cifti import extract_cifti_scalar_data
from modelarrayio.utils.fixels import nifti2_to_mif
from modelarrayio.utils.voxels import flattened_image


def summarize_rows(rows) -> dict[str, np.ndarray]:
    """Compute common diagnostics from a sequence of 1-D subject arrays."""
    stacked = np.vstack(rows)
    return {
        'mean': np.nanmean(stacked, axis=0).astype(np.float32),
        'n_non_nan': np.sum(~np.isnan(stacked), axis=0).astype(np.float32),
        'element_id': np.arange(stacked.shape[1], dtype=np.float32),
    }


def verify_nifti_element_mapping(group_mask_img, group_mask_matrix):
    """Verify NIfTI group-mask flattening order matches element indices."""
    expected = np.arange(int(group_mask_matrix.sum()), dtype=np.float32)
    element_volume = np.zeros(group_mask_matrix.shape, dtype=np.float32)
    element_volume[group_mask_matrix] = expected
    element_img = nb.Nifti1Image(
        element_volume,
        affine=group_mask_img.affine,
        header=group_mask_img.header,
    )
    extracted = flattened_image(element_img, group_mask_img, group_mask_matrix)
    if not np.array_equal(extracted.astype(np.int64), expected.astype(np.int64)):
        raise ValueError('Element ID mapping check failed for NIfTI group-mask flattening.')


def write_nifti_diagnostics(
    *,
    maps: list[str],
    scalar_name: str,
    diagnostics: dict[str, np.ndarray],
    group_mask_img,
    group_mask_matrix,
    output_dir: Path,
):
    header = group_mask_img.header.copy()
    header.set_data_dtype(np.float32)
    for name in maps:
        out_file = output_dir / f'{scalar_name}_{name}.nii.gz'
        data = np.zeros(group_mask_matrix.shape, dtype=np.float32)
        data[group_mask_matrix] = diagnostics[name]
        nb.Nifti1Image(data, affine=group_mask_img.affine, header=header).to_filename(out_file)


def verify_cifti_element_mapping(template_cifti, reference_brain_names):
    """Verify CIFTI extraction order matches element indices."""
    expected = np.arange(reference_brain_names.shape[0], dtype=np.float32)
    test_img = nb.Cifti2Image(
        expected.reshape(1, -1),
        header=template_cifti.header,
        nifti_header=template_cifti.nifti_header,
    )
    recovered, _ = extract_cifti_scalar_data(test_img, reference_brain_names=reference_brain_names)
    if not np.array_equal(recovered.astype(np.int64), expected.astype(np.int64)):
        raise ValueError('Element ID mapping check failed for CIFTI greyordinate ordering.')


def write_cifti_diagnostics(
    *,
    maps: list[str],
    scalar_name: str,
    diagnostics: dict[str, np.ndarray],
    template_cifti,
    output_dir: Path,
):
    for name in maps:
        out_file = output_dir / f'{scalar_name}_{name}.dscalar.nii'
        nb.Cifti2Image(
            diagnostics[name].reshape(1, -1),
            header=template_cifti.header,
            nifti_header=template_cifti.nifti_header,
        ).to_filename(out_file)


def verify_mif_element_mapping(template_nifti2, num_elements: int):
    """Verify fixel vector reshape/squeeze mapping remains identity."""
    expected = np.arange(num_elements, dtype=np.float32)
    test_img = nb.Nifti2Image(
        expected.reshape(-1, 1, 1),
        affine=template_nifti2.affine,
        header=template_nifti2.header,
    )
    recovered = test_img.get_fdata(dtype=np.float32).squeeze()
    if not np.array_equal(recovered.astype(np.int64), expected.astype(np.int64)):
        raise ValueError('Element ID mapping check failed for MIF fixel vector ordering.')


def write_mif_diagnostics(
    *,
    maps: list[str],
    scalar_name: str,
    diagnostics: dict[str, np.ndarray],
    template_nifti2,
    output_dir: Path,
):
    for name in maps:
        out_file = output_dir / f'{scalar_name}_{name}.mif'
        temp_nifti2 = nb.Nifti2Image(
            diagnostics[name].reshape(-1, 1, 1),
            affine=template_nifti2.affine,
            header=template_nifti2.header,
        )
        nifti2_to_mif(temp_nifti2, out_file)
