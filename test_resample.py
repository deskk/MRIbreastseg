import SimpleITK as sitk

mri_pre_path = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external-duke-fgt/duke_outputs/phase0/Breast_MRI_640/Breast_MRI_640_PRE_registered.nii.gz"
mri_dyn_crop_path = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external-duke-fgt/duke_outputs/phase5_split_mri/left/Breast_MRI_640/Breast_MRI_640_DYN_cropped.nii.gz"

img_pre = sitk.ReadImage(mri_pre_path)
img_dyn_crop = sitk.ReadImage(mri_dyn_crop_path)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(img_dyn_crop)
resampler.SetInterpolator(sitk.sitkBSpline)
resampler.SetDefaultPixelValue(0)

img_pre_crop = resampler.Execute(img_pre)
print("Resampled PRE shape:", img_pre_crop.GetSize())
