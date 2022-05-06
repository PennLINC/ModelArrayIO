# how to use ConVoxel

## Prepare csv file, for example:
"cohort_FA.csv" for scalar FA:
| ***scalar_name*** | ***source_file***  | ***source_mask_file***  | subject_id    | age    | sex     | 
| :----:        | :----:         | :----:         | :----:        | :----: |  :----: |
| FA            | FA/sub1_FA.mif | FA/sub1_FA_mask.mif | sub1          | 10     | F       |
| FA            | FA/sub2_FA.mif | FA/sub2_FA_mask.mif | sub2          | 20     | M       |
| FA            | FA/sub3_FA.mif | FA/sub3_FA_mask.mif | sub3          | 15     | F       |
| ...            | ... | ... | ...          | ...     | ...       |

Notes:
* Columns that must be included are highlighted in ***bold and italics***;
    * Notice that compared to csv file for ConFixel, here we need to provide a column called `source_mask_file`. This column is for subject-specific masks, i.e. the boundary of the subject-specific images can be different from the group mask. If you don't have subject-specific masks, simply provide group mask here for each subject.
* The order of columns can be changed.


## ConVoxel --> volume data

All the converted volume data are saved with data type float32. 

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