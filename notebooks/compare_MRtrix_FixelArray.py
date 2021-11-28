import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from plotnine import ggplot, aes, geom_point, geom_text

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

f_mrtrix_p_permu = folder_mrtrix + "/uncorrected_pvalue_F1.mif"
f_mrtrix_p_permu_fwe = folder_mrtrix + "/fwe_1mpvalue_F1.mif"

_, fixelarray_raw_p = mif_to_nifti2(f_fixelarray_raw_p)
_, fixelarray_p_fdr = mif_to_nifti2(f_fixelarray_p_fdr)
_, fixelarray_p_bonferroni = mif_to_nifti2(f_fixelarray_p_bonferroni)

_, mrtrix_p_permu = mif_to_nifti2(f_mrtrix_p_permu)
mrtrix_p_permu = 1 - mrtrix_p_permu
_, mrtrix_p_permu_fwe = mif_to_nifti2(f_mrtrix_p_permu_fwe)
mrtrix_p_permu_fwe = 1 - mrtrix_p_permu_fwe


# # save to text file:
# temp_fn = open(f_fixelarray_raw_p.replace(".mif",'.txt'), "w")
# np.savetxt(temp_fn, fixelarray_raw_p)
# temp_fn.close()
#
# temp_fn = open(f_fixelarray_p_fdr.replace(".mif",'.txt'), "w")
# np.savetxt(temp_fn, fixelarray_p_fdr)
# temp_fn.close()
#
# temp_fn = open(f_fixelarray_p_bonferroni.replace(".mif",'.txt'), "w")
# np.savetxt(temp_fn, fixelarray_p_bonferroni)
# temp_fn.close()
#
# temp_fn = open(f_mrtrix_p_permu.replace(".mif",'.txt'), "w")
# np.savetxt(temp_fn, mrtrix_p_permu)
# temp_fn.close()
#
# temp_fn = open(f_mrtrix_p_permu_fwe.replace(".mif",'.txt'), "w")
# np.savetxt(temp_fn, mrtrix_p_permu_fwe)
# temp_fn.close()


pearsonr(fixelarray_raw_p, mrtrix_p_permu)
pearsonr(fixelarray_p_fdr, mrtrix_p_permu)
pearsonr(fixelarray_p_fdr, mrtrix_p_permu_fwe)
pearsonr(fixelarray_p_bonferroni, mrtrix_p_permu_fwe)

d = {'fixelarray_raw_p': fixelarray_raw_p,
     'fixelarray_p_fdr': fixelarray_p_fdr,
     'fixelarray_p_bonferroni': fixelarray_p_bonferroni,
     'mrtrix_p_permu': mrtrix_p_permu,
     'mrtrix_p_permu_fwe': mrtrix_p_permu_fwe} 

df = pd.DataFrame(data=d)


str_title = "corr=" + str(pearsonr(fixelarray_p_fdr, mrtrix_p_permu)[0])

(
    ggplot(df)   # What data to use
    + aes(x="fixelarray_p_fdr", y="mrtrix_p_permu")  # What variable to use
    +  geom_point()
    # + ggtitle(str)
    
)

(
    ggplot(df)  # What data to use
    + aes(x="fixelarray_raw_p", y="mrtrix_p_permu")  # What variable to use
    + geom_point()
)

(
    ggplot(df)  # What data to use
    + aes(x="fixelarray_p_fdr", y="mrtrix_p_permu_fwe")  # What variable to use
    + geom_point()
)

(
    ggplot(df)  # What data to use
    + aes(x="fixelarray_p_bonferroni", y="mrtrix_p_permu_fwe")  # What variable to use
    + geom_point()
)


print()




