import sys

sys.path.insert(0, '../confixel')

from fixels import * 

folder_fixelarray = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/results/ltn_FDC_n938_wResults_nfixel-0_20210908-135041"
folder_mrtrix = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/for_fixelcfestats/stats_FDC/nsubj-938.nthreads-1.ftests.notest.nshuffles-100.interactive.runMemProfiler.s-1sec.20210908-133419"

# Fstats:
f1 = folder_fixelarray + "/lm_model.statistic.mif"
f2 = folder_mrtrix + "/Fvalue_F1.mif"

# slope:
#f1 = folder_fixelarray + "/lm_Age.estimate.mif"
#f2 = folder_mrtrix + "/beta1.mif"

_, from_fixelarray = mif_to_nifti2(f1)
_, from_mrtrix = mif_to_nifti2(f2)

diffabsmax = max(abs(from_fixelarray - from_mrtrix))
print(diffabsmax)
