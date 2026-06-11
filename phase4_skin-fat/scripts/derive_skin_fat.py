import os
import glob
import json
import numpy as np
import SimpleITK as sitk

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
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
    phase1_left_dir = config["PHASE1"]["OUTPUT_LEFT_DIR"]
    phase1_right_dir = config["PHASE1"]["OUTPUT_RIGHT_DIR"]
    fat_dir = config["PHASE4"]["OUTPUT_FAT_DIR"]
    skin_dir = config["PHASE4"]["OUTPUT_SKIN_DIR"]
    radius = config["PHASE4"]["SKIN_EROSION_RADIUS"]
    
    sides_dirs = {
        'left': phase1_left_dir,
        'right': phase1_right_dir
    }

    for side, side_dir in sides_dirs.items():
        if not os.path.exists(side_dir):
            continue
            
        subjects = [d for d in os.listdir(side_dir) if os.path.isdir(os.path.join(side_dir, d))]
        target_label = 1 if side == 'left' else 2
        
        for subj in subjects:
            subj_dir = os.path.join(side_dir, subj)
            bd_files = glob.glob(os.path.join(subj_dir, "*_BreastDivider_Mask.nii.gz"))
            if not bd_files:
                continue
                
            print(f"Phase 4 [Skin-Fat] Processing {subj} ({side})...")
            bd_img = sitk.ReadImage(bd_files[0])
            bd_arr = sitk.GetArrayFromImage(bd_img)
            
            # 1. Generate Fat Mask (Everything in the BreastDivider boundary)
            fat_arr = np.zeros_like(bd_arr, dtype=np.uint8)
            fat_arr[bd_arr == target_label] = 1
            
            fat_out_dir = os.path.join(fat_dir, side, subj)
            os.makedirs(fat_out_dir, exist_ok=True)
            fat_img = sitk.GetImageFromArray(fat_arr)
            fat_img.CopyInformation(bd_img)
            sitk.WriteImage(fat_img, os.path.join(fat_out_dir, f"{subj}_fat_mask.nii.gz"))
            
            # 2. Generate Skin Mask
            skin_arr = protect_posterior_and_erode(bd_arr, target_label, radius)
            
            skin_out_dir = os.path.join(skin_dir, side, subj)
            os.makedirs(skin_out_dir, exist_ok=True)
            skin_img = sitk.GetImageFromArray(skin_arr)
            skin_img.CopyInformation(bd_img)
            sitk.WriteImage(skin_img, os.path.join(skin_out_dir, f"{subj}_skin_mask.nii.gz"))

if __name__ == "__main__":
    main()
