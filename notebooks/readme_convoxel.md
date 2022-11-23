# how to use ConVoxel

In general, `ConVoxel` is very similar to `ConFixel`.

## Prepare data
To convert (a list of) voxel-wise data from NIfTI format to .h5 format, you need to prepare a cohort CSV file that provides several basic informations of all NIfTI files you want to include. We recommend that, for each scalar (e.g. FA), prepare one .csv file, and thus getting one .h5 file.

In addition, different from `ConFixel`, you also need to provide these image files:
* one group mask: Only voxels within the group mask will be kept during conversion to .h5 file.
* subject-specific masks: This takes the inconsistent boundary of subject-specific images into account. After conversion, for each subject's scalar mage, voxels outside the subject-specific mask will be set to `NaN`. `ModelArray` will then check if each voxel has sufficient number of subjects to get reliable statistics (see argument `num.subj.lthr.abs` and `num.subj.lthr.rel` in Model fitting functions, e.g., [`ModelArray.lm()`](https://pennlinc.github.io/ModelArray/reference/ModelArray.lm.html)).
    * If you don't have subject-specific masks, that's fine; you can use group mask instead (see below for how to achieve this in .csv file).

### Cohort's csv file (for each scalar)
Each row of a cohort .csv is for one NIfTI file you want to include. The file should at least include these columns (Notes: these column names are fixed, i.e. not user-defined):

* "scalar_name": which tells us what metric is being analyzed, e.g. FA
* "source_file": which tells us which NIfTI file will be used for this subject
* "source_mask_file": which tells us the filename of subject-specific masks. If you don't have subject-specific masks, simply provide filename of group mask here for each subject.

### Example
#### Example folder structure
```
/home/username/myProject/data
|
├── cohort_FA.csv
├── group_mask.nii.gz
│
├── FA
|   ├── sub1_FA.nii.gz
|   ├── sub2_FA.nii.gz
|   ├── sub3_FA.nii.gz
│   ├── ...
│
├── individual_masks
|   ├── sub1_mask.nii.gz
|   ├── sub2_mask.nii.gz
|   ├── sub3_mask.nii.gz
|   ├── ...
└── ...
```

#### Corresponding csv file for scalar FA can look like this:
"cohort_FA.csv" for scalar FA:
| ***scalar_name*** | ***source_file***  | ***source_mask_file***  | subject_id    | age    | sex     |
| :----:        | :----:         | :----:         | :----:        | :----: |  :----: |
| FA            | FA/sub1_FA.nii.gz | individual_masks/sub1_mask.nii.gz | sub1          | 10     | F       |
| FA            | FA/sub2_FA.nii.gz | individual_masks/sub2_mask.nii.gz | sub2          | 20     | M       |
| FA            | FA/sub3_FA.nii.gz | individual_masks/sub3_mask.nii.gz | sub3          | 15     | F       |
| ...            | ... | ... | ...          | ...     | ...       |

Notes:
* Columns that must be included are highlighted in ***bold and italics***;
* The order of columns can be changed.

For this case, when running ConVoxel, argument `--relative-root` should be `/home/username/myProject/data`

## Run ConVoxel
### Convert NIfTI files to an HDF5 (.h5) file
Using above described scenario as an example, for FA dataset:
``` console
foo@bar:~$ # first, activate conda environment where `ConFixel` is installed: `conda activate <env_name>`
foo@bar:~$ convoxel \
                --group-mask-file group_mask.nii.gz \
                --cohort-file cohort_FA.csv \
                --relative-root /home/username/myProject/data \
                --output-hdf5 FA.h5
```

Now you should get the HDF5 file "FA.h5" in folder "/home/username/myProject/data". You may use [`ModelArray`](https://pennlinc.github.io/ModelArray/) to perform statistical analysis.

### Convert result .h5 file back to NIfTI format
After running `ModelArray` and getting statistical results in FA.h5 file (say, the analysis name is called "mylm"), you can use `volumestats_write` to convert results into a list of NIfTI files in a folder specified by you.

``` console
foo@bar:~$ # first, activate conda environment where `ConFixel` is installed: `conda activate <env_name>`
foo@bar:~$ volumestats_write \
                --group-mask-file group_mask.nii.gz \
                --cohort-file cohort_FA.csv \
                --relative-root /home/username/myProject/data \
                --analysis-name mylm \
                --input-hdf5 FA.h5 \
                --output-dir FA_stats \
                --output-ext .nii.gz    # or ".nii"
```

Now you should get the results NIfTI images saved in folder "FA_stats". All the converted volume data are saved with data type float32.

### For additional information:
You can refer to `--help` for additional information:
``` console
foo@bar:~$ convoxel --help
foo@bar:~$ volumestats_write --help
```

## Other notes
### Image of number of observations used
If you requested `nobs` when running model fitting in `ModelArray`, after conversion back to NIfTI files, you'll get an image called `*_model.nobs.nii*` (number of observations used). With the feature of `subject-specific masks`, you'll probably see inhomogeneity in this image.

### Results for voxels without sufficient subjects (because of subject-specific masks):


| software | regular voxel  | voxel without sufficient subjects  |
| :----:        | :----:         | :----:         |
| python nibabel | nobs: e.g. 209.0; p.value: e.g. 0.010000...  |  nobs or p.value or 1m.p.value: nan |
| MRtrix mrview | nobs: e.g. 209; p.value: e.g. 0.010000| nobs or p.value or 1m.p.value: ?|
| ITK-snap | nobs: e.g. 209; p.value: e.g. 0.0100 | nobs or p.value or 1m.p.value: 0 |

Notes:
* `regular voxel` means voxels with sufficient subjects
* Here `nobs` is number of observations from ModelArray, which should be integer. Also, expect it is not constant across voxels if different subject-specific masks were provided
* Although ITK-snap does not display NaN as "?", those voxels without sufficient subjects won't pop out when thresholding p.values (e.g. p.value ranging 0-0.05, 1m.p.value ranging 0.95-1), so they won't be "mixed" with other regular voxels.