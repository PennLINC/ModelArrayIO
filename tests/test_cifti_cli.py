import os
import os.path as op
import csv
import subprocess
import sys

import numpy as np
import nibabel as nb
from nibabel.cifti2.cifti2_axes import ScalarAxis, BrainModelAxis
import h5py


def _make_synthetic_cifti_dscalar(mask_bool: np.ndarray, values: np.ndarray) -> nb.Cifti2Image:
    # Build axes: single scalar and a brain model from a volumetric mask
    scalar_axis = ScalarAxis(["synthetic"])  # one scalar map
    brain_axis = BrainModelAxis.from_mask(mask_bool)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    # Data must be 2D: (nmaps, ngrayordinates)
    data_2d = values.reshape(1, -1).astype(np.float32)
    return nb.Cifti2Image(data_2d, header=header)


def test_concifti_cli_creates_expected_hdf5(tmp_path):
    # Create a small volumetric mask for brain model axis
    vol_shape = (3, 3, 3)
    mask = np.zeros(vol_shape, dtype=bool)
    true_vox = [(0, 0, 0), (0, 1, 2), (1, 1, 1), (2, 2, 0), (2, 1, 2)]
    for ijk in true_vox:
        mask[ijk] = True
    n_grayordinates = int(mask.sum())

    # Create two subjects with simple sequences
    subjects = []
    for sidx in range(2):
        vals = np.arange(n_grayordinates, dtype=np.float32) + sidx
        img = _make_synthetic_cifti_dscalar(mask, vals)
        path = tmp_path / f"sub-{sidx+1}.dscalar.nii"
        img.to_filename(path)
        subjects.append(str(path.name))

    # Build cohort CSV
    cohort_csv = tmp_path / "cohort_cifti.csv"
    with cohort_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scalar_name", "source_file"])
        writer.writeheader()
        for sname in subjects:
            writer.writerow({
                "scalar_name": "THICK",
                "source_file": sname,
            })

    out_h5 = tmp_path / "out_cifti.h5"
    cmd = [
        sys.executable,
        "-m",
        "modelarrayio.cifti",
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
    assert proc.returncode == 0, f"concifti failed: {proc.stdout}\n{proc.stderr}"
    assert op.exists(out_h5)

    # Validate HDF5 contents
    with h5py.File(out_h5, "r") as h5:
        assert "greyordinates" in h5
        grey = np.array(h5["greyordinates"])  # stored as transposed table (2, N)
        assert grey.shape[0] == 2  # vertex_id, structure_id
        n = grey.shape[1]
        assert n == n_grayordinates

        # structure_names present
        g = h5["greyordinates"]
        assert "structure_names" in g.attrs
        struct_names = g.attrs["structure_names"]
        assert len(struct_names) >= 1

        # Scalars dataset
        dset = h5["scalars/THICK/values"]
        num_subjects, num_items = dset.shape
        assert num_subjects == 2
        assert num_items == n_grayordinates

        # Column names exist and match subjects count
        grp = h5["scalars/THICK"]
        assert "column_names" in grp
        colnames = list(map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x), grp["column_names"][...]))
        assert len(colnames) == 2

        # Spot-check a couple values
        assert np.isclose(float(dset[0, 0]), 0.0)
        assert np.isclose(float(dset[1, 0]), 1.0)


