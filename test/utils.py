"""Shared test utility functions for ModelArrayIO tests."""

from __future__ import annotations

import nibabel as nb
import numpy as np
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ParcelsAxis, ScalarAxis


def make_parcels_axis(parcel_names: list[str]) -> ParcelsAxis:
    """Create a minimal surface-only ParcelsAxis (one vertex per parcel on the left cortex)."""
    n = len(parcel_names)
    nvertices = {'CIFTI_STRUCTURE_CORTEX_LEFT': n}
    vox_dtype = np.dtype([('ijk', '<i4', (3,))])
    voxels = [np.array([], dtype=vox_dtype) for _ in range(n)]
    vertices = [{'CIFTI_STRUCTURE_CORTEX_LEFT': np.array([i], dtype=np.int32)} for i in range(n)]
    return ParcelsAxis(parcel_names, voxels, vertices, np.eye(4), (10, 10, 10), nvertices)


def make_dscalar(mask_bool: np.ndarray, values: np.ndarray) -> nb.Cifti2Image:
    """Create a synthetic dscalar CIFTI image from a volumetric boolean mask and values."""
    scalar_axis = ScalarAxis(['synthetic'])
    brain_axis = BrainModelAxis.from_mask(mask_bool)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    return nb.Cifti2Image(values.reshape(1, -1).astype(np.float32), header=header)


def make_pscalar(parcel_names: list[str], values: np.ndarray) -> nb.Cifti2Image:
    """Create a synthetic pscalar CIFTI image with the given parcel names and values."""
    scalar_axis = ScalarAxis(['synthetic'])
    parcels_axis = make_parcels_axis(parcel_names)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, parcels_axis))
    return nb.Cifti2Image(values.reshape(1, -1).astype(np.float32), header=header)


def make_pconn(parcel_names: list[str], values: np.ndarray) -> nb.Cifti2Image:
    """Create a synthetic pconn CIFTI image; values are reshaped to (n_parcels, n_parcels)."""
    parcels_axis = make_parcels_axis(parcel_names)
    header = nb.cifti2.Cifti2Header.from_axes((parcels_axis, parcels_axis))
    n = len(parcel_names)
    return nb.Cifti2Image(values.reshape(n, n).astype(np.float32), header=header)


def make_nifti(data: np.ndarray, affine: np.ndarray | None = None) -> nb.Nifti1Image:
    """Create a Nifti1Image with the given data and an optional affine (default: identity)."""
    if affine is None:
        affine = np.eye(4)
    return nb.Nifti1Image(data.astype(np.float32), affine)
