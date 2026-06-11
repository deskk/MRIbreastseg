import os
import glob
import sys
import numpy as np
import SimpleITK as sitk

DATASET_PATH = "/local/scratch/scratch-hd/desmond/dataset/clean_data_registered"
NNUNET_RAW_TEST = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/nnUNet/nnunetv2/nnUNet_raw/Dataset102_Test/imagesTs"

os.makedirs(NNUNET_RAW_TEST, exist_ok=True)


def get_dynamic_sequences(subject_dir):
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    
    pre_file = None
    post_files = []
    
    for f in all_files:
        filename = os.path.basename(f).lower()
        if "sub" in filename or "mask" in filename or "t2" in filename or ("t1" in filename and "dynamic" not in filename and "dyanamic" not in filename):
            continue
            
        if "pre" in filename:
            pre_file = f
        elif "dynamic" in filename or "dyanamic" in filename:
            post_files.append(f)
            
    # Sort post files by the digit prefix "sXXXXXX"
    post_files.sort(key=lambda x: int(os.path.basename(x).split(" ")[0][1:]))
    
    return pre_file, post_files

def zscore_normalization_sitk(image_sitk, mean, std):
    array = sitk.GetArrayFromImage(image_sitk)
    normalized_array = (array - mean) / std
    zscored_sitk = sitk.GetImageFromArray(normalized_array)
    zscored_sitk.CopyInformation(image_sitk)
    return zscored_sitk

def resample_sitk(image_sitk, new_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkBSpline, tol=0.00001):
    original_size = image_sitk.GetSize()
    original_spacing = image_sitk.GetSpacing()
   
    if len(original_size) == 2:
        original_size = original_size + (1, )
    if len(original_spacing) == 2:
        original_spacing = original_spacing + (1.0, )

    new_size = [round(original_size[0]*(original_spacing[0] + tol) / new_spacing[0]),
                round(original_size[1]*(original_spacing[1] + tol) / new_spacing[1]),
                round(original_size[2]*(original_spacing[2] + tol) / new_spacing[2])]

    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(new_spacing)
    ResampleFilter.SetSize(np.array(new_size, dtype='int').tolist())
    ResampleFilter.SetOutputDirection(image_sitk.GetDirection())
    ResampleFilter.SetOutputOrigin(image_sitk.GetOrigin())
    ResampleFilter.SetOutputPixelType(image_sitk.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    try:
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)
    except RuntimeError:
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ResampleFilter.SetOutputDirection(direction)
        image_sitk.SetDirection(direction)
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)

    return resampled_image_sitk

def preprocess_subject(subject_id):
    subject_dir = os.path.join(DATASET_PATH, subject_id)
    print(f"\n--- Processing subject {subject_id} ---")
    
    pre_file, post_files = get_dynamic_sequences(subject_dir)
    print(f"Found PRE: {os.path.basename(pre_file) if pre_file else 'None'}")
    print(f"Found POST ({len(post_files)} phases): {[os.path.basename(p) for p in post_files]}")
    
    if not post_files:
        print("ERROR: No post-contrast sequences found.")
        return
        
    first_post_file = post_files[0]
    all_dce_files = post_files.copy()
    if pre_file:
        all_dce_files.append(pre_file)
        
    # global z-score calculation
    print("Calculating global Z-score...")
    all_pixels = []
    first_post_sitk = None
    
    for f in all_dce_files:
        img_sitk = sitk.ReadImage(f, sitk.sitkFloat32)
        if f == first_post_file:
            first_post_sitk = img_sitk
            
        arr = sitk.GetArrayFromImage(img_sitk).ravel()
        arr = arr[arr != 0] # exclude background 0
        all_pixels.append(arr)
        
    all_pixels = np.concatenate(all_pixels)
    global_mean = np.mean(all_pixels)
    global_std = np.std(all_pixels)
    print(f"Global Mean: {global_mean:.4f}, Global Std: {global_std:.4f}")
    
    # normalize first post phase
    print(f"Applying normalization to 1st POST ({os.path.basename(first_post_file)})...")
    normalized_sitk = zscore_normalization_sitk(first_post_sitk, global_mean, global_std)
    
    # isotropic resampling
    print("Resampling to 1x1x1 isotropic spacing...")
    resampled_sitk = resample_sitk(normalized_sitk, new_spacing=[1.0, 1.0, 1.0])
    
    output_path = os.path.join(NNUNET_RAW_TEST, f"MAMAMIA_{subject_id}_0000.nii.gz")
    sitk.WriteImage(resampled_sitk, output_path)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    all_dirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    for subj in sorted(all_dirs):
        preprocess_subject(subj)
