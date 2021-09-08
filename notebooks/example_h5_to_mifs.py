from ../confixel/fixels import *    # h5_to_mifs

flag_whichdataset = "josiane"    # "val" or "test_n50" or "josiane"

if flag_whichdataset == "test_n50" :
    example_mif = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/fixeldata/sub1_afd.mif"
    h5_file = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/n50_fixels_output_test.h5"
    analysis_name = "lm"
    fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_forCircleCI_n50/h5_to_mifs"

elif flag_whichdataset == "val" :
    example_mif = "/home/chenying/Desktop/fixel_project/data/data_from_Val/FixelMask_PopulationTemplate/FD/sub-ZAPR01C105-FD.mif"
    # h5_file = "/home/chenying/Desktop/fixel_project/data/data_from_Val/fixels_analysis_copy.h5"
    # analysis_name = "lm"
    # fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_from_Val/h5_to_mifs"
    
    # h5_file = "/home/chenying/Desktop/fixel_project/data/data_from_Val/fixels_analysis_copy_correctedp.h5"
    # analysis_name = "lm_correctedp"
    # fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_from_Val/h5_to_mifs"
    
    h5_file = "/home/chenying/Desktop/fixel_project/data/data_from_Val/fixels_28yrAndBelow_only1_analysis_copy.h5"
    analysis_name = "lm"
    fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_from_Val/h5_to_mifs_28yrAndBelow_only1"
elif flag_whichdataset == "josiane" :
    example_mif = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/for_fixelcfestats/fdc_10_smoothed_10fwhm_new/sub-80010.mif"
    h5_file = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/results/ltn_FDC_n938_wResults_nfixel-0_20210908-135041.h5"
    analysis_name = "lm"
    fixel_output_dir = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/results/ltn_FDC_n938_wResults_nfixel-0_20210908-135041"

h5_to_mifs(example_mif, h5_file, analysis_name, fixel_output_dir)

# h5_data = h5py.File(h5_file, "r")
#
# # h5_file['results/' + analysis_name + '/results_matrix'].attrs['colnames']
# results_matrix = h5_data['results/' + analysis_name + '/results_matrix']
#
# names_data = results_matrix.attrs['colnames']
#
#
# print()