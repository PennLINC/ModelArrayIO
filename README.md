# ConFixel
`ConFixel` is companion converter software for [ModelArray](https://pennlinc.github.io/ModelArray/) for converting data back and forth from the HDF5 file format.

<p align="center">

![Overview](overview_structure.PNG)

</p>

`ConFixel` software includes two converters: `ConFixel` for fixel-wise data (`.mif`), and `ConVoxel` for voxel-wise data (NIfTI). Each converter converts between the original image format and the HDF5 file format (`.h5`) that ModelArray uses. 

Below lists the commands in each converter. After [installation](#installation), these commands can be directly called in a terminal console.

* `ConFixel` converter for fixel-wise data (MRtrix image format `.mif`):
    * `.mif` --> `.h5`: command `confixel`
    * `.h5` --> `.mif`: command `fixelstats_write`
* `ConVoxel` converter for voxel-wise data (NIfTI):
    * NIfTI --> `.h5`: command `convoxel`
    * `.h5` --> NIfTI: command `volumestats_write`

## Installation
### Install dependent software MRtrix (only required for fixel-wise data `.mif`)
When converting fixel-wise data's format (`.mif`), converter `ConFixel` uses function `mrconvert` from MRtrix, so please make sure MRtrix has been installed. If it's not installed yet, please refer to [MRtrix's webpage](https://www.mrtrix.org/download/) for how to install it. Type `mrview` in the terminal to check whether MRtrix installation is successful.

If your input data is voxel-wise data, you can skip this step.

### Install `ConFixel` software
Before installing ConFixel software, you may want to create a conda environment  - see [here](https://pennlinc.github.io/ModelArray/articles/installations.html) for more. If you installed MRtrix in a conda environment, you can directly install ConFixel software in that environment.

You can install `ConFixel` software from [GitHub](https://github.com/PennLINC/ConFixel):

``` console
git clone https://github.com/PennLINC/ConFixel.git
cd ConFixel
pip install .   # for end user

# you may remove the original source code if you are an end user:
cd ..
rm -r ConFixel
```
If you are a developer, and if there is any update in the source code locally, you may update the installation with:
``` console
# Supporse you're in root directory of ConFixel source code:
pip install -e .    # for developer to update
```

## How to use
We provide [walkthrough of how to use `ConFixel` for fixel-wise data](notebooks/walkthrough_fixel-wise_data.md), and [walkthrough of `ConVoxel` for voxel-wise data](notebooks/walkthrough_voxel-wise_data.md).

As `ConFixel` software is usually used together with [ModelArray](https://pennlinc.github.io/ModelArray/), we also provide [a combined walkthrough](https://pennlinc.github.io/ModelArray/articles/walkthrough.html) of ConFixel + ModelArray with example fixel-wise data.

You can also refer to `--help` for additional information:
``` console 
confixel --help
```
You can replace `confixel` with other commands in ConFixel.
