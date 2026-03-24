########################################
Using ModelArrayIO from the command line
########################################


********
confixel
********

.. argparse::
   :ref: modelarrayio.fixels:get_parser
   :prog: confixel
   :func: get_parser


********
convoxel
********

.. argparse::
   :ref: modelarrayio.voxels:get_parser
   :prog: convoxel
   :func: get_parser


********
concifti
********

.. argparse::
   :ref: modelarrayio.cifti:get_parser
   :prog: concifti
   :func: get_parser


****************
fixelstats_write
****************

.. argparse::
   :ref: modelarrayio.fixels:get_h5_to_fixels_parser
   :prog: fixelstats_write
   :func: get_h5_to_fixels_parser

*****************
volumestats_write
*****************

.. argparse::
   :ref: modelarrayio.voxels:get_h5_to_volume_parser
   :prog: volumestats_write
   :func: get_h5_to_volume_parser


****************
ciftistats_write
****************

.. argparse::
   :ref: modelarrayio.cifti:get_h5_to_ciftis_parser
   :prog: ciftistats_write
   :func: get_h5_to_ciftis_parser
