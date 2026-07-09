import os
import glob
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def get_skin_mask(bd_arr, radius):
    """
    Finds the posterior-most chest wall envelope using y_max and extrapolates it
    to bury the chest wall before 3D erosion, leaving the anterior folds exposed.
    """
    from scipy.ndimage import label
    fg = ((bd_arr == 1) | (bd_arr == 2)).astype(np.uint8)
    Z, Y, X = fg.shape
    
    # 1. Clean fg to find the true chest wall surface robustly using scipy (Fast LCC)
    labels, num_features = label(fg)
    if num_features > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0 # Ignore background
        max_label = sizes.argmax()
        fg_clean = (labels == max_label).astype(np.uint8)
    else:
        fg_clean = fg
        
    H = np.full((Z, X), -1, dtype=int)
    for z in range(Z):
        for x in range(X):
            y_indices = np.where(fg_clean[z, :, x])[0]
            if len(y_indices) > 0:
                H[z, x] = y_indices[-1] # y_max
                
    invalid = (H == -1)
    if np.any(~invalid):
        indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        H_extrapolated = H[indices[0], indices[1]]
    else:
        H_extrapolated = H
        
    padded_arr = np.zeros_like(bd_arr, dtype=np.uint8)
    padded_arr[fg == 1] = 1
    
    for z in range(Z):
        for x in range(X):
            y_max = H_extrapolated[z, x]
            if y_max != -1:
                padded_arr[z, y_max:, x] = 1
                
    padded_img = sitk.GetImageFromArray(padded_arr)
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelRadius(radius)
    eroder.SetKernelType(sitk.sitkBall)
    eroder.SetBoundaryToForeground(True)
    
    eroded_padded_img = eroder.Execute(padded_img == 1)
    eroded_arr = sitk.GetArrayFromImage(eroded_padded_img)
    
    skin_arr = np.zeros_like(bd_arr, dtype=np.uint8)
    skin_arr[(fg == 1) & (eroded_arr == 0)] = 1
    
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
        # if os.path.exists(expected_fat):
        #     print(f"Phase 4 [Skin-Fat] Skipping {subj}, output already exists.")
        #     continue
            
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
        
        # 2. Generate Skin Mask (Combined logic)
        skin_combined_arr = get_skin_mask(bd_arr, radius)
        
        skin_out_dir = os.path.join(skin_dir, subj)
        os.makedirs(skin_out_dir, exist_ok=True)
        skin_img = sitk.GetImageFromArray(skin_combined_arr)
        skin_img.CopyInformation(bd_img)
        sitk.WriteImage(skin_img, os.path.join(skin_out_dir, f"{subj}_skin_mask.nii.gz"))

if __name__ == "__main__":
    main()
