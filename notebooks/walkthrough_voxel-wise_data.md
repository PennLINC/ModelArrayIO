# Walkthrough for voxel-wise data conversion

For voxel-wise data, we use converter `ConVoxel`. In general, `ConVoxel` is very similar to converter `ConFixel`.

## Prepare data
To convert (a list of) voxel-wise data from NIfTI format to .h5 format, you need to prepare a cohort CSV file that provides several basic informations of all NIfTI files you want to include. We recommend that, for each scalar (e.g. FA), prepare one .csv file, and thus getting one .h5 file.

In addition, different from converter `ConFixel`, you also need to provide these image files:
* one group mask: Only voxels within the group mask will be kept during conversion to .h5 file.
* subject-specific masks (i.e., individual masks): This takes the inconsistent boundary of subject-specific images into account. After conversion, for each subject's scalar mage, voxels outside the subject-specific mask will be set to `NaN`. `ModelArray` will then check if each voxel has sufficient number of subjects to get reliable statistics (see argument `num.subj.lthr.abs` and `num.subj.lthr.rel` in Model fitting functions, e.g., [`ModelArray.lm()`](https://pennlinc.github.io/ModelArray/reference/ModelArray.lm.html)).
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
|   ├── sub-01_FA.nii.gz
|   ├── sub-02_FA.nii.gz
|   ├── sub-03_FA.nii.gz
│   ├── ...
│
├── individual_masks
|   ├── sub-01_mask.nii.gz
|   ├── sub-02_mask.nii.gz
|   ├── sub-03_mask.nii.gz
|   ├── ...
└── ...
```

#### Corresponding csv file for scalar FA can look like this:
"cohort_FA.csv" for scalar FA:
| ***scalar_name*** | ***source_file***  | ***source_mask_file***  | subject_id    | age    | sex     |
| :----:        | :----:         | :----:         | :----:        | :----: |  :----: |
| FA            | FA/sub-01_FA.nii.gz | individual_masks/sub-01_mask.nii.gz | sub-01          | 10     | F       |
| FA            | FA/sub-02_FA.nii.gz | individual_masks/sub-02_mask.nii.gz | sub-02          | 20     | M       |
| FA            | FA/sub-03_FA.nii.gz | individual_masks/sub-03_mask.nii.gz | sub-03          | 15     | F       |
| ...            | ... | ... | ...          | ...     | ...       |

Notes:
* Columns that must be included are highlighted in ***bold and italics***;
* The order of columns can be changed.
* Please make sure the consistency (see examples as below). They are case sensitive, too! Either upper case or lower case works, just need to be consistent :)
    * Folder name e.g., "FA" in column `source_file` in CSV file v.s. the actual folder name on disk;
    * File names in columns `source_file` and `source_mask_file` in CSV file v.s. the actual file names on disk;
    * Scalar name e.g., "FA" in column `scalar_name` in CSV file v.s. what you will specify when using functions in `ModelArray`;

For this case, when running ConVoxel, argument `--relative-root` should be `/home/username/myProject/data`

## Run ConVoxel
### Convert NIfTI files to an HDF5 (.h5) file
Using above described scenario as an example, for FA dataset:
``` console
# first, activate conda environment where `ConFixel` is installed: `conda activate <env_name>`
convoxel \
    --group-mask-file group_mask.nii.gz \
    --cohort-file cohort_FA.csv \
    --relative-root /home/username/myProject/data \
    --output-hdf5 FA.h5
```

Now you should get the HDF5 file "FA.h5" in folder "/home/username/myProject/data". You may use [`ModelArray`](https://pennlinc.github.io/ModelArray/) to perform statistical analysis.

### Convert result .h5 file back to NIfTI format
After running `ModelArray` and getting statistical results in FA.h5 file (say, the analysis name is called "mylm"), you can use `volumestats_write` to convert results into a list of NIfTI files in a folder specified by you.

``` console
# first, activate conda environment where software `ConFixel` is installed: `conda activate <env_name>`
volumestats_write \
    --group-mask-file group_mask.nii.gz \
    --cohort-file cohort_FA.csv \
    --relative-root /home/username/myProject/data \
    --analysis-name mylm \
    --input-hdf5 FA.h5 \
    --output-dir FA_stats \
    --output-ext .nii.gz    # or ".nii"
```

Now you should get the results NIfTI images saved in folder "FA_stats". All the converted volume data are saved with data type float32. You can view the images with the image viewer you like.

> ⚠️ ⚠️ WARNING ⚠️ ⚠️ : See [notes regarding "Existing output folder and output images"](#existing-output-folder-and-output-images).

### For additional information:
You can refer to `--help` for additional information:
``` console
convoxel --help
volumestats_write --help
```

## Other notes
### ConVoxel: convert from `.h5` to NIfTI
#### Existing output folder and output images
⚠️ ⚠️ WARNING ⚠️ ⚠️ 
* If the output folder already exists, `ConVoxel` will not delete it or create a new one. You will only get a message saying "WARNING: Output directory exists". Therefore, if there were existing files in the output folder, and they are not part of the current list of images to be saved (e.g., results to be saved were changed, but the output folder name was not changed), these files will be kept as it is and won't be deleted.   <!--- confirmed with toy data, 3/9/2023 -->
    * However, for existing files which are still part of the current list to be saved, they will be replaced. This is different from current implementation of `ConFixel` converter for fixel-wise data.   <!--- confirmed with toy data, 3/9/2023 -->
* So to avoid confusion and better for version controls, if the output folder already exists, you might consider manually deleting it before using `ConVoxel` to save new images.


### Image of number of observations used
If you requested `nobs` when running model fitting in `ModelArray`, after conversion back to NIfTI files, you'll get an image called `*_model.nobs.nii*` (number of observations used). With the feature of "subject-specific masks", you'll probably see inhomogeneity in this image.

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