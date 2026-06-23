import SimpleITK as sitk
import os

uu_file = '/local/scratch/scratch-hd/desmond/datasets/UU/clean_data/505656/s000007 AX T1 3D FS DYNAMIC.nii.gz'
duke_file = '/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external/duke_outputs/phase0/Breast_MRI_002/Breast_MRI_002_DYN_registered.nii.gz'

uu_img = sitk.ReadImage(uu_file)
duke_img = sitk.ReadImage(duke_file)

print('UU DYN:')
print('  Size:', uu_img.GetSize())
print('  Spacing:', uu_img.GetSpacing())
print('  Direction:', uu_img.GetDirection())
print('  Origin:', uu_img.GetOrigin())

print('\nDuke DYN:')
print('  Size:', duke_img.GetSize())
print('  Spacing:', duke_img.GetSpacing())
print('  Direction:', duke_img.GetDirection())
print('  Origin:', duke_img.GetOrigin())
