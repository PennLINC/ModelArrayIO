#!/bin/bash
which_dataset=$1

fn_py="../confixel/fixels.py"
# input data:
if [[ "$which_dataset" == "data_forCircleCI_n50" ]]
then
	index_file="FD/index.mif"
	directions_file="FD/directions.mif"
	cohort_file="n50_cohort.csv"
	output_hdf5="fixels.h5"
	relative_root="../../data/data_forCircleCI_n50"
	
elif [[ "$which_dataset" == "data_forCircleCI_n25x2" ]]
then
	index_file="fixeldata/index.mif"
	directions_file="fixeldata/directions.mif"
	cohort_file="n25x2_cohort.csv"
	output_hdf5="fixels_withSubjID.h5"
	relative_root="../data/data_forCircleCI_n25x2"
fi



cmd="python $fn_py"
cmd+=" --index-file ${index_file}"
cmd+=" --directions-file ${directions_file}"
cmd+=" --cohort-file ${cohort_file}"
cmd+=" --output-hdf5 ${output_hdf5}"
cmd+=" --relative-root ${relative_root}"

echo $cmd
$cmd
echo ""
