#!/bin/bash
which_dataset=$1

fn_py="../FixelDB/inst/python/fixeldb/create_backend/fixels.py"
# input data:
if [[ "$which_dataset" == "data_from_Val" ]]
then
	index_file="fixeldata/index.mif"
	directions_file="fixeldata/directions.mif"
	cohort_file="test_cohort.csv"
	output_hdf5="fixels.h5"
	relative_root="../data/data_forCircleCI_n=2"
elif [[ "$which_dataset" == "data_forCircleCI_n50" ]]
then
	index_file="fixeldata/index.mif"
	directions_file="fixeldata/directions.mif"
	cohort_file="test_cohort_n50.csv"
	colname_subjid="subject_id"
	output_hdf5="fixels.h5"
	relative_root="../data/data_forCircleCI_n50"
elif [[ "$which_dataset" == "data_forCircleCI_n25x2" ]]
then
	index_file="fixeldata/index.mif"
	directions_file="fixeldata/directions.mif"
	cohort_file="n25x2_cohort.csv"
	colname_subjid="subject_id"
	output_hdf5="fixels_withSubjID.h5"
	relative_root="../data/data_forCircleCI_n25x2"
elif [[ "$which_dataset" == "data_from_Val" ]]
then
	#index_file="fixeldata/index.mif"
	index_file="FixelMask_PopulationTemplate/index.mif"
	#directions_file="fixeldata/directions.mif"
	directions_file="FixelMask_PopulationTemplate/directions.mif"
	cohort_file="cohort_ZAPR01.csv"
	output_hdf5="fixels_wRealSubjID.h5"
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
cmd+=" --colname_subjid ${colname_subjid}"
cmd+=" --output-hdf5 ${output_hdf5}"
cmd+=" --relative-root ${relative_root}"

echo $cmd
$cmd
echo ""
