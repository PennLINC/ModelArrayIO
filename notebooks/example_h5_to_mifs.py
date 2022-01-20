import sys
import os
#sys.path.insert(0,'../confixel/')
sys.path.append( os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "confixel"  ))

from confixel.fixels import *    # h5_to_mifs

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
flag_whichdataset = "josiane"    # "test_n50" or "josiane"
flag_option = 2  # 1 for h5_to_mifs(), 2 for console command fixelstats_write
    # before running flag_option=2 & flag_whichdataset="josiane": need to change folder "for_fixelcfestats" (in "data_from_josiane") to "FDC"
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if flag_option == 1:
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

elif flag_option == 2:   ## using console commands + h5_to_fixels():
    
    if flag_whichdataset == "josiane" :
        # temp: can be entered as args in vscode's launch.json; also need to change main(): calling h5_to_fixels()
        temp = ["--index-file", "results/ltn_FDC_n938_wResults_nfixels-0_20220109-183909/index.mif",
                "--directions-file", "results/ltn_FDC_n938_wResults_nfixels-0_20220109-183909/directions.mif",
                "--cohort-file", "df_example_n938.csv",
                "--relative-root", "/home/chenying/Desktop/fixel_project/data/data_from_josiane",
                "--analysis-name", "gam_allOutputs",
                "--input-hdf5", "results/ltn_FDC_n938_wResults_nfixels-0_20220109-183909.h5",
                "--output-dir", "results/output_test"]
        cmd = 'fixelstats_write ' + ' '.join(temp)    
        os.system(cmd)
