import os
import os.path as op
import csv
import subprocess
import sys

import numpy as np
import nibabel as nb
import h5py


def _make_nifti(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    return nb.Nifti1Image(data.astype(np.float32), affine)


def _ijk_value(i, j, k):
    return i * 100.0 + j * 10.0 + k * 1.0


def test_convoxel_cli_creates_expected_hdf5(tmp_path):
    # Small synthetic volume
    shape = (5, 6, 7)
    group_mask = np.zeros(shape, dtype=bool)
    # Create a sparse pattern of true voxels
    true_coords = [(0, 1, 1), (1, 2, 3), (2, 4, 5), (3, 0, 0), (4, 5, 6), (1, 1, 4), (2, 2, 2)]
    for (i, j, k) in true_coords:
        group_mask[i, j, k] = True

    # Save group mask
    group_mask_img = _make_nifti(group_mask.astype(np.uint8))
    group_mask_file = tmp_path / "group_mask.nii.gz"
    group_mask_img.to_filename(group_mask_file)

    # Create two subjects with individual masks (one drops a voxel)
    subjects = []
    for sidx in range(2):
        # Scalar volume encodes f(i,j,k)
        scalar = np.zeros(shape, dtype=np.float32)
        for (i, j, k) in true_coords:
            scalar[i, j, k] = _ijk_value(i, j, k) + sidx  # slight per-subject shift

        # Individual mask: subject 1 omits one voxel
        indiv_mask = group_mask.copy()
        if sidx == 1:
            omit = true_coords[1]
            indiv_mask[omit] = False

        scalar_img = _make_nifti(scalar)
        mask_img = _make_nifti(indiv_mask.astype(np.uint8))

        scalar_path = tmp_path / f"sub-{sidx+1}_scalar.nii.gz"
        mask_path = tmp_path / f"sub-{sidx+1}_mask.nii.gz"
        scalar_img.to_filename(scalar_path)
        mask_img.to_filename(mask_path)
        subjects.append((str(scalar_path.name), str(mask_path.name)))

    # Build cohort CSV (relative paths)
    cohort_csv = tmp_path / "cohort.csv"
    with cohort_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scalar_name", "source_file", "source_mask_file"])
        writer.writeheader()
        for sidx, (scalar_name, mask_name) in enumerate(subjects):
            writer.writerow({
                "scalar_name": "FA",
                "source_file": scalar_name,
                "source_mask_file": mask_name,
            })

    # Run CLI using module to avoid PATH issues
    out_h5 = tmp_path / "out.h5"
    cmd = [
        sys.executable,
        "-m",
        "modelarrayio.voxels",
        "--group-mask-file", str(group_mask_file.name),
        "--cohort-file", str(cohort_csv.name),
        "--relative-root", str(tmp_path),
        "--output-hdf5", str(out_h5.name),
        "--backend", "hdf5",
        "--dtype", "float32",
        "--compression", "gzip",
        "--compression-level", "1",
        "--shuffle", "True",
        "--chunk-voxels", "0",
        "--target-chunk-mb", "1.0",
    ]
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"convoxel failed: {proc.stdout}\n{proc.stderr}"
    assert op.exists(out_h5)

    # Validate HDF5 contents
    with h5py.File(out_h5, "r") as h5:
        assert "voxels" in h5
        vox = np.array(h5["voxels"])  # stored as transposed table (3, N)
        assert vox.shape[0] == 3
        ijk = np.vstack(np.nonzero(group_mask))  # (3, N) ordered by i, then j, then k
        assert vox.shape[1] == ijk.shape[1]

        # Check ordering matches nonzero order (allow exact match)
        assert np.array_equal(vox, ijk)

        # Scalars dataset
        dset = h5["scalars/FA/values"]
        num_subjects, num_voxels = dset.shape
        assert num_subjects == 2
        assert num_voxels == ijk.shape[1]

        # Column names exist and match subjects count
        grp = h5["scalars/FA"]
        assert "column_names" in grp
        colnames = list(map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x), grp["column_names"][...]))
        assert len(colnames) == 2

        # Spot-check a voxel mapping (pick the third voxel)
        vidx = 2
        i, j, k = int(ijk[0, vidx]), int(ijk[1, vidx]), int(ijk[2, vidx])
        expected_s0 = _ijk_value(i, j, k) + 0
        expected_s1 = _ijk_value(i, j, k) + 1
        # If subject 1 omitted that voxel, it should be NaN (masked out becomes NaN on flatten)
        v0 = float(dset[0, vidx])
        v1 = float(dset[1, vidx])
        assert np.isclose(v0, expected_s0, equal_nan=True)
        # Determine whether subject 1 omitted this voxel
        omitted = False
        omit = true_coords[1]
        if (i, j, k) == omit:
            omitted = True
        if omitted:
            assert np.isnan(v1)
        else:
            assert np.isclose(v1, expected_s1, equal_nan=True)


