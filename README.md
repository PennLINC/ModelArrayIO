# ConFixel: Moved to ModelArrayIO

ConFixel has moved beyond fixels and handles a lot of other modalities.
It also got optimized!
[Going forward, ConFixel is now ModelArrayIO.](https://pennlinc.github.io/ModelArrayIO/). 
This repository will be left here because the URL is in the publication.


`ConFixel` is companion converter software for [ModelArray](https://pennlinc.github.io/ModelArray/) for converting data back and forth from the HDF5 file format.

<p align="center">

![Overview](overview_structure.png)

</p>

`ConFixel` software includes three converters: `ConFixel` for fixel-wise data (`.mif`), `ConVoxel` for voxel-wise data (NIfTI) and `ConCIFTI` for CIFTI-2 dscalar files. Each converter converts between the original image format and the HDF5 file format (`.h5`) that ModelArray uses.

Below lists the commands in each converter. After [installation](#installation), these commands can be directly called in a terminal console.

* `ConFixel` converter for fixel-wise data (MRtrix image format `.mif`):
    * `.mif` --> `.h5`: command `confixel`
    * `.h5` --> `.mif`: command `fixelstats_write`
* `ConVoxel` converter for voxel-wise data (NIfTI):
    * NIfTI --> `.h5`: command `convoxel`
    * `.h5` --> NIfTI: command `volumestats_write`
* `ConCIFTI` converter for greyordinate-wise data (CIFTI-2):
    * CIFTI-2 --> `.h5`: command `concifti`
    * `.h5` --> CIFTI-2: command `ciftistats_write`

## Installation
### Install dependent software MRtrix (only required for fixel-wise data `.mif`)
When converting fixel-wise data's format (`.mif`), converter `ConFixel` uses function `mrconvert` from MRtrix, so please make sure MRtrix has been installed. If it's not installed yet, please refer to [MRtrix's webpage](https://www.mrtrix.org/download/) for how to install it. Type `mrview` in the terminal to check whether MRtrix installation is successful.

If your input data is voxel-wise data or CIFTI (greyordinate-wise) data, you can skip this step.

### Install `ConFixel` software
Before installing ConFixel software, you may want to create a conda environment  - see [here](https://pennlinc.github.io/ModelArray/articles/installations.html) for more. If you installed MRtrix in a conda environment, you can directly install ConFixel software in that environment.

You can install `ConFixel` software from `GitHub`:

``` console
git clone https://github.com/PennLINC/ConFixel.git
cd ConFixel
pip install .   # build via pyproject.toml
```

If you are a developer, and if there is any update in the source code locally, you may update the installation with an editable install:

``` console
# From the repository root
pip install -e .
```

Alternatively, if you have `hatch` installed, you can build wheels/sdist locally:

``` console
hatch build
pip install dist/*.whl
```

## How to use
We provide [walkthrough for how to use `ConFixel` for fixel-wise data](notebooks/walkthrough_fixel-wise_data.md), and [walkthrough for `ConVoxel` for voxel-wise data](notebooks/walkthrough_voxel-wise_data.md).

As `ConFixel` software is usually used together with [ModelArray](https://pennlinc.github.io/ModelArray/), we also provide [a combined walkthrough](https://pennlinc.github.io/ModelArray/articles/walkthrough.html) of ConFixel + ModelArray with example fixel-wise data.

You can also refer to `--help` for additional information:
``` console
confixel --help
```
You can replace `confixel` with other commands in ConFixel.

## Storage backends: HDF5 and TileDB

ModelArrayIO supports two on-disk backends for the subject-by-element matrix:

- HDF5 (default), implemented in `modelarrayio/h5_storage.py`
- TileDB, implemented in `modelarrayio/tiledb_storage.py`

Both backends expose a similar API:

- create a dense 2D array `(subjects, items)` and write all values at once
- create an empty array with the same shape and write by column stripes
- write/read column names alongside the data

Notes and minor differences:
- Chunking vs tiling: HDF5 uses chunks; TileDB uses tiles. We compute tile sizes analogous to chunk sizes to keep write/read patterns similar.
- Compression: HDF5 uses `gzip` by default; TileDB defaults to `zstd`+shuffle for better speed/ratio. You can switch to `gzip` for parity.
- Metadata: HDF5 stores `column_names` as a dataset attribute; TileDB stores names as JSON metadata on the array/group.
- Layout: Both backends keep dimensions in the same order and use zero-based indices.
