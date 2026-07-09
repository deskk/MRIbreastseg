import os
import json
import SimpleITK as sitk
import numpy as np

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config_duke.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config = load_config()
    
    out_mri_left = config["PHASE5"]["OUTPUT_SPLIT_MRI_LEFT_DIR"]
    fusion_left_dir = config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"]
    fusion_left_native_dir = fusion_left_dir + "_native"
    
    os.makedirs(fusion_left_native_dir, exist_ok=True)
    
    subjects = [d for d in os.listdir(fusion_left_dir) if os.path.isdir(os.path.join(fusion_left_dir, d))]
    
    print(f"Generating native resolution masks for {len(subjects)} subjects...")
    
    for i, subj in enumerate(subjects):
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(subjects)} subjects...")
            
        mri_path = os.path.join(out_mri_left, subj, f"{subj}_DYN_cropped.nii.gz")
        mask_path = os.path.join(fusion_left_dir, subj, f"{subj}_final_fusion.nii.gz")
        
        if not os.path.exists(mri_path) or not os.path.exists(mask_path):
            continue
            
        mri_img = sitk.ReadImage(mri_path)
        mask_img = sitk.ReadImage(mask_path)
        
        # Resample the 0.25mm mask back onto the native MRI grid
        # using NearestNeighbor interpolation to preserve integer class labels
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mri_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        
        native_mask = resampler.Execute(mask_img)
        
        out_subj_dir = os.path.join(fusion_left_native_dir, subj)
        os.makedirs(out_subj_dir, exist_ok=True)
        sitk.WriteImage(native_mask, os.path.join(out_subj_dir, f"{subj}_final_fusion_native.nii.gz"))

    print("Native masks generated successfully!")

if __name__ == "__main__":
    main()
