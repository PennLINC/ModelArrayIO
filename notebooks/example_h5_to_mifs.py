import sys
sys.path.insert(0,'../confixel/')

from fixels import *    # h5_to_mifs

flag_whichdataset = "test_n50"    # "test_n50" or "josiane"

if flag_whichdataset == "test_n50" :
    example_mif = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/fixeldata/sub1_afd.mif"
    h5_file = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/n50_fixels_output_test.h5"
    analysis_name = "lm"
    fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/h5_to_mifs"
elif flag_whichdataset == "josiane" :
    example_mif = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/for_fixelcfestats/fdc_10_smoothed_10fwhm_new/sub-80010.mif"
    h5_file = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/results/ltn_FDC_n938_wResults_nfixel-0_20211126-182543.h5"
    analysis_name = "gam_allOutputs"
    fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/results/ltn_FDC_n938_wResults_nfixel-0_20211126-182543"

h5_to_mifs(example_mif, h5_file, analysis_name, fixel_output_dir)
