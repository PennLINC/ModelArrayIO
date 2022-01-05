# ConFixel
Convert Fixels back and forth from h5 files!

## Prepare data
To convert (a list of) fixel data from .mif format to .h5 format, you need to prepare a cohort csv file that provides several basic informations of all .mif files you want to include. We recommend that, for each scalar (e.g. FD/FC/FDC), prepare one .csv file, and thus getting one .h5 file.

### Cohort's csv file (for each scalar)
Each row of a cohort .csv is for one mif file you want to include. The file should at least include these columns (Notes: these column names are fixed, i.e. not user-defined):

* "scalar_name": name of fixel data metric, e.g. FD, FC, or FDC 
* "source_file": the path to the mif file. This should include both the filename itself (e.g. sub1_fd.mif), but also any necessary relative path to this file (e.g. If a mif file is in folder "/home/username/myProject/FD", and when you call python script "fixels.py" to run the conversion, you specify --relative-root as "/home/username/myProject/", then, source_file should be "FD/sub1_fd.mif" for this mif file). See example below for more.

### Example
#### Example folder organization

#### Example csv file for a scalar:


## Run the conversion

### From .mif to .h5
Example please see "notebooks/example_mif_to_h5.sh"

### From .h5 to .mif
Example please see "notebooks/example_h5_to_mifs.py"