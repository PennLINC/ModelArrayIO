import sys
import os
#sys.path.insert(0,'../confixel/')
sys.path.append( os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "confixel"  ))

from confixel.fixels import *    # h5_to_mifs

flag_whichdataset = "josiane"    # "test_n50" or "josiane"

if flag_whichdataset == "test_n50" :
    example_mif = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/FD/sub1_fd.mif"
    h5_file = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/n50_fixels_output_test.h5"
    analysis_name = "lm"
    fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/test_outputs"
elif flag_whichdataset == "josiane" :
    folder_main = "/home/chenying/Desktop/fixel_project/data/data_from_josiane"
    tag_results = "ltn_FDC_n938_wResults_nfixels-0_20220109-183909"

    example_mif = os.path.join(folder_main, "for_fixelcfestats/fdc_10_smoothed_10fwhm_new/sub-80010.mif")
    h5_file = os.path.join(folder_main, "results", tag_results+".h5")
    analysis_name = "gam_allOutputs"
    fixel_output_dir = os.path.join(folder_main, "results",tag_results)

h5_to_mifs(example_mif, h5_file, analysis_name, fixel_output_dir)
