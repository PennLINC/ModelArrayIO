from fixels import * 
example_mif = "/home/chenying/Desktop/fixel_project/data/data_from_josiane/index.mif"

nifti2_img, _ = mif_to_nifti2(example_mif)

temp_nifti2 = nb.Nifti2Image([1,1,1,1],
                                     nifti2_img.affine,
                                     header=nifti2_img.header)
