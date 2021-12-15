#!/bin/bash
which_dataset=$1

fn_py="../FixelDB/inst/python/fixeldb/create_backend/fixels.py"
# input data:
if [[ "$which_dataset" == "data_forCircleCI_n=2" ]]
then
	index_file="fixeldata/index.mif"
	directions_file="fixeldata/directions.mif"
	cohort_file="test_cohort.csv"
	output_hdf5="fixels.h5"
	relative_root="../data/data_forCircleCI_n=2"
elif [[ "$which_dataset" == "data_forCircleCI_n=50" ]]
then
	index_file="fixeldata/index.mif"
	directions_file="fixeldata/directions.mif"
	cohort_file="test_cohort_n50.csv"
	output_hdf5="fixels.h5"
	relative_root="../data/data_forCircleCI_n50"
elif [[ "$which_dataset" == "data_from_Val" ]]
then
	#index_file="fixeldata/index.mif"
	index_file="FixelMask_PopulationTemplate/index.mif"
	#directions_file="fixeldata/directions.mif"
	directions_file="FixelMask_PopulationTemplate/directions.mif"
	cohort_file="cohort_ZAPR01.csv"
	output_hdf5="fixels.h5"
	relative_root="../data/data_from_Val"
elif [[ "$which_dataset" == "data_from_Val_28yrAndBelow_only1" ]]
then
	index_file="FixelMask_PopulationTemplate/index.mif"
	directions_file="FixelMask_PopulationTemplate/directions.mif"
	cohort_file="cohort_ZAPR01_28yrAndBelow_only1.csv"
	output_hdf5="fixels_28yrAndBelow_only1.h5"
	relative_root="../data/data_from_Val"
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
