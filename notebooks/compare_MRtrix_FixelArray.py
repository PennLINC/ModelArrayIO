import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from plotnine import ggplot, aes, geom_point

from confixel.fixels import mif_to_nifti2

sys.path.insert(0, '../confixel')

from fixels import * 

folder_fixelarray = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/results/ltn_FDC_n938_wResults_nfixel-0_20210908-135041"
#folder_mrtrix = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/for_fixelcfestats/stats_FDC/nsubj-938.nthreads-1.ftests.notest.nshuffles-100.interactive.runMemProfiler.s-1sec.20210908-133419"
folder_mrtrix = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/for_fixelcfestats/stats_FDC/nsubj-938.nthreads-4.ftests.nshuffles-100.vmware.runMemProfiler.s-1sec.20210909-210227"

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

# model's p:
f_fixelarray_raw_p = folder_fixelarray + "/lm_model.p.value.mif"
f_fixelarray_p_fdr = folder_fixelarray + "/lm_model.p.value.fdr.mif"
f_fixelarray_p_bonferroni = folder_fixelarray + "/lm_model.p.value.bonferroni.mif"

f_mrtrix_uncorrected_p = folder_mrtrix + "/uncorrected_pvalue_F1.mif"
f_mrtrix_fwe_p = folder_mrtrix + "/fwe_1mpvalue_F1.mif"

_, fixelarray_raw_p = mif_to_nifti2(f_fixelarray_raw_p)
_, fixelarray_p_fdr = mif_to_nifti2(f_fixelarray_p_fdr)
_, fixelarray_p_bonferroni = mif_to_nifti2(f_fixelarray_p_bonferroni)

_, mrtrix_uncorrected_p = mif_to_nifti2(f_mrtrix_uncorrected_p)
mrtrix_uncorrected_p = 1 - mrtrix_uncorrected_p
_, mrtrix_fwe_p = mif_to_nifti2(f_mrtrix_fwe_p)
mrtrix_fwe_p = 1 - mrtrix_fwe_p

pearsonr(fixelarray_raw_p, mrtrix_uncorrected_p)
pearsonr(fixelarray_p_fdr, mrtrix_uncorrected_p)
pearsonr(fixelarray_p_fdr, mrtrix_fwe_p)


d = {'fixelarray_raw_p': fixelarray_raw_p,
     'fixelarray_p_fdr': fixelarray_p_fdr,
     'fixelarray_p_bonferroni': fixelarray_p_bonferroni,
     'mrtrix_uncorrected_p': mrtrix_uncorrected_p,
     'mrtrix_fwe_p': mrtrix_fwe_p} 

df = pd.DataFrame(data=d)

(
    ggplot(df)  # What data to use
    + aes(x="fixelarray_p_fdr", y="mrtrix_uncorrected_p")  # What variable to use
    + geom_point()
)

(
    ggplot(df)  # What data to use
    + aes(x="fixelarray_p_fdr", y="mrtrix_fwe_p")  # What variable to use
    + geom_point()
)


print()




