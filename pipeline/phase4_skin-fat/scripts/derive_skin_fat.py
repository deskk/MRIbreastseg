import os
import glob
import json
import numpy as np
import SimpleITK as sitk

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def protect_posterior_and_erode(mask_arr, target_label, radius):
    """
    Pads the posterior direction (positive Y) to infinity to prevent 
    the 3D morphological erosion from eroding the chest wall, thereby
    preventing a 'hollow shell' and creating a proper skin envelope.
    """
    padded_arr = mask_arr.copy()
    
    # Simulate an infinitely deep chest wall towards the posterior (+Y)
    for z in range(padded_arr.shape[0]):
        for x in range(padded_arr.shape[2]):
            y_indices = np.where(padded_arr[z, :, x] == target_label)[0]
            if len(y_indices) > 0:
                y_max = y_indices[-1]
                padded_arr[z, y_max:, x] = target_label
                
    padded_img = sitk.GetImageFromArray(padded_arr)
    
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelRadius(radius)
    eroder.SetKernelType(sitk.sitkBall)
    
    # Protect straight, flat boundaries of the image from erosion
    eroder.SetBoundaryToForeground(True)
    
    eroded_padded_img = eroder.Execute(padded_img == target_label)
    eroded_arr = sitk.GetArrayFromImage(eroded_padded_img)
    
    # Skin is original foreground MINUS eroded padded foreground
    skin_arr = np.zeros_like(mask_arr)
    skin_arr[(mask_arr == target_label) & (eroded_arr == 0)] = 1
    return skin_arr

def main():
    config = load_config()
    phase1_mask_dir = config["PHASE1"]["OUTPUT_MASK_DIR"]
    fat_dir = config["PHASE4"]["OUTPUT_FAT_DIR"]
    skin_dir = config["PHASE4"]["OUTPUT_SKIN_DIR"]
    radius = config["PHASE4"]["SKIN_EROSION_RADIUS"]
    
    if not os.path.exists(phase1_mask_dir):
        print(f"Phase 1 mask dir not found: {phase1_mask_dir}")
        return
        
    subjects = [d for d in os.listdir(phase1_mask_dir) if os.path.isdir(os.path.join(phase1_mask_dir, d))]
    
    test_subjects = config.get("TEST_SUBJECTS", [])
    if test_subjects:
        subjects = [s for s in subjects if s in test_subjects]
    
    for subj in subjects:
        expected_fat = os.path.join(fat_dir, subj, f"{subj}_fat_mask.nii.gz")
        if os.path.exists(expected_fat):
            print(f"Phase 4 [Skin-Fat] Skipping {subj}, output already exists.")
            continue
            
        subj_dir = os.path.join(phase1_mask_dir, subj)
        bd_files = glob.glob(os.path.join(subj_dir, "*_BreastDivider_Mask.nii.gz"))
        if not bd_files:
            continue
            
        print(f"Phase 4 [Skin-Fat] Processing full torso {subj}...")
        bd_img = sitk.ReadImage(bd_files[0])
        bd_arr = sitk.GetArrayFromImage(bd_img)
        
        # 1. Generate Fat Mask (Everything in the BreastDivider boundary, labels 1 and 2)
        fat_arr = np.zeros_like(bd_arr, dtype=np.uint8)
        fat_arr[bd_arr == 1] = 1
        fat_arr[bd_arr == 2] = 1
        
        fat_out_dir = os.path.join(fat_dir, subj)
        os.makedirs(fat_out_dir, exist_ok=True)
        fat_img = sitk.GetImageFromArray(fat_arr)
        fat_img.CopyInformation(bd_img)
        sitk.WriteImage(fat_img, os.path.join(fat_out_dir, f"{subj}_fat_mask.nii.gz"))
        
        # 2. Generate Skin Mask (Combine left and right skin)
        skin_left_arr = protect_posterior_and_erode(bd_arr, 1, radius)
        skin_right_arr = protect_posterior_and_erode(bd_arr, 2, radius)
        
        skin_combined_arr = np.zeros_like(bd_arr, dtype=np.uint8)
        skin_combined_arr[skin_left_arr == 1] = 1
        skin_combined_arr[skin_right_arr == 1] = 1
        
        skin_out_dir = os.path.join(skin_dir, subj)
        os.makedirs(skin_out_dir, exist_ok=True)
        skin_img = sitk.GetImageFromArray(skin_combined_arr)
        skin_img.CopyInformation(bd_img)
        sitk.WriteImage(skin_img, os.path.join(skin_out_dir, f"{subj}_skin_mask.nii.gz"))

if __name__ == "__main__":
    main()
