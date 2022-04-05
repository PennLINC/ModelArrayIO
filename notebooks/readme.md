# Detailed descriptions
## ConFixel: h5 to mif files
* If there are already existing images in the output folder, the images won't be overwritten. This is because in funciton confixel.fixels.nifti2_to_mif, we did not turn on -force in "mrconvert".