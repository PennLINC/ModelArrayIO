"""Unit tests for MIF/fixel utility helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from modelarrayio.cli.h5_to_mif import h5_to_mif
from modelarrayio.cli.mif_to_h5 import mif_to_h5
from modelarrayio.utils import mif
from modelarrayio.utils.misc import load_and_normalize_cohort


@pytest.mark.downloaded_data
def test_mif_to_h5_results(
    tmp_path_factory: pytest.TempPathFactory, downloaded_fixel_data_dir: Path
) -> None:
    """Test mif-to-h5 and h5-to-mif conversion, mimicking a ModelArray analysis."""
    import os

    # Step 1: Prepare inputs for conversion
    out_dir = tmp_path_factory.mktemp('data_fixel_toy')
    in_dir = downloaded_fixel_data_dir
    index_file = in_dir / 'index.mif'
    directions_file = in_dir / 'directions.mif'
    cohort_file = in_dir / 'stat-alpha_cohort.csv'
    if not index_file.exists():
        raise FileNotFoundError(f'Contents of {in_dir}:\n{os.listdir(in_dir)}')

    # Prepend absolute path to source files in cohort file
    cohort_df = pd.read_csv(cohort_file)
    cohort_df['source_file'] = cohort_df['source_file'].map(lambda path: str(in_dir / path))
    temp_cohort_file = out_dir / 'stat-alpha_cohort.csv'
    cohort_df.to_csv(temp_cohort_file, index=False)
    cohort_long, _ = load_and_normalize_cohort(temp_cohort_file)

    # Step 2: Convert MIF to HDF5
    h5_file = out_dir / 'stat-alpha.h5'
    assert (
        mif_to_h5(
            index_file=index_file,
            directions_file=directions_file,
            cohort_long=cohort_long,
            output=h5_file,
        )
        == 0
    )

    # Step 3: Add a result (element-wise mean across files) to the HDF5 file
    with h5py.File(h5_file, 'a') as h5:
        alpha_values = h5['scalars/alpha/values'][...]
        mean_values = np.mean(alpha_values, axis=0, dtype=np.float32)
        results_group = h5.require_group('results/lm')
        results_group.create_dataset('results_matrix', data=mean_values[np.newaxis, :])
        results_group.create_dataset(
            'column_names',
            data=np.array(['mean'], dtype=h5py.string_dtype('utf-8')),
        )

    # Step 4: Convert HDF5 to MIF
    output_dir = out_dir / 'mif_results'
    assert (
        h5_to_mif(
            example_mif=cohort_df['source_file'].iloc[0],
            in_file=h5_file,
            analysis_name='lm',
            compress=False,
            output_dir=output_dir,
        )
        == 0
    )

    # Step 5: Calculate mean directly from MIF files
    source_arrays = [
        mif.mif_to_image(source_file)[1].astype(np.float32)
        for source_file in cohort_df['source_file']
    ]
    direct_mean = np.mean(np.stack(source_arrays, axis=0), axis=0, dtype=np.float32)

    # Step 6: Compare the mean from the HDF5 file to the mean from the MIF files
    output_mif = output_dir / 'lm_mean.mif'
    assert output_mif.exists()

    result_img, result_data = mif.mif_to_image(output_mif)
    template_img, _ = mif.mif_to_image(cohort_df['source_file'].iloc[0])

    assert isinstance(result_img, mif.MifImage)
    np.testing.assert_allclose(result_img.affine, template_img.affine)
    np.testing.assert_allclose(result_data, direct_mean)
