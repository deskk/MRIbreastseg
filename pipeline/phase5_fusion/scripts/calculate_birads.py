import os
import glob
import csv
import json
import numpy as np
import SimpleITK as sitk
import pandas as pd

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def calculate_birads(fgt_ratio):
    if pd.isna(fgt_ratio) or fgt_ratio is None:
        return None
    if fgt_ratio < 0.25:
        return 'A'
    elif fgt_ratio < 0.50:
        return 'B'
    elif fgt_ratio < 0.75:
        return 'C'
    else:
        return 'D'

def get_fgt_ratios(fusion_path):
    if not os.path.exists(fusion_path):
        return None, None, None
    
    img = sitk.ReadImage(fusion_path)
    arr = sitk.GetArrayFromImage(img)
    
    # Masks
    breast_mask = (arr == 1) | (arr == 3)
    fgt_mask = (arr == 3)
    
    # Volumetric
    fat_pixels = np.sum(arr == 1)
    fgt_pixels = np.sum(fgt_mask)
    total_vol = fat_pixels + fgt_pixels
    vol_ratio = float(fgt_pixels) / float(total_vol) if total_vol > 0 else 0.0
    
    # Axial MIP (axis 0)
    axial_breast = np.any(breast_mask, axis=0)
    axial_fgt = np.any(fgt_mask, axis=0)
    axial_breast_area = np.sum(axial_breast)
    axial_fgt_area = np.sum(axial_fgt)
    axial_ratio = float(axial_fgt_area) / float(axial_breast_area) if axial_breast_area > 0 else 0.0
    
    # Sagittal MIP (axis 2)
    sagittal_breast = np.any(breast_mask, axis=2)
    sagittal_fgt = np.any(fgt_mask, axis=2)
    sagittal_breast_area = np.sum(sagittal_breast)
    sagittal_fgt_area = np.sum(sagittal_fgt)
    sagittal_ratio = float(sagittal_fgt_area) / float(sagittal_breast_area) if sagittal_breast_area > 0 else 0.0
    
    return vol_ratio, axial_ratio, sagittal_ratio

def main():
    config = load_config()
    left_dir = config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"] + "_native"
    right_dir = config["PHASE5"]["OUTPUT_FUSION_RIGHT_DIR"] + "_native"
    
    # Use unified summary CSV
    summary_csv = config["PHASE5"]["OUTPUT_SUMMARY_CSV"]
    
    if not os.path.exists(summary_csv):
        print(f"Missing summary CSV! Expected at {summary_csv}")
        return
        
    df = pd.read_csv(summary_csv, dtype={'Subject': str})
    
    results = {
        'Left_Vol_Ratio': [], 'Left_Axial_Ratio': [], 'Left_Sagittal_Ratio': [],
        'Left_Vol_Class': [], 'Left_Axial_Class': [], 'Left_Sagittal_Class': [],
        'Right_Vol_Ratio': [], 'Right_Axial_Ratio': [], 'Right_Sagittal_Ratio': [],
        'Right_Vol_Class': [], 'Right_Axial_Class': [], 'Right_Sagittal_Class': []
    }
    
    for _, row in df.iterrows():
        subj = row['Subject']
        
        left_path = os.path.join(left_dir, subj, f"{subj}_final_fusion.nii.gz")
        right_path = os.path.join(right_dir, subj, f"{subj}_final_fusion.nii.gz")
        
        l_vol, l_ax, l_sag = get_fgt_ratios(left_path)
        r_vol, r_ax, r_sag = get_fgt_ratios(right_path)
        
        results['Left_Vol_Ratio'].append(l_vol)
        results['Left_Axial_Ratio'].append(l_ax)
        results['Left_Sagittal_Ratio'].append(l_sag)
        results['Left_Vol_Class'].append(calculate_birads(l_vol))
        results['Left_Axial_Class'].append(calculate_birads(l_ax))
        results['Left_Sagittal_Class'].append(calculate_birads(l_sag))
        
        results['Right_Vol_Ratio'].append(r_vol)
        results['Right_Axial_Ratio'].append(r_ax)
        results['Right_Sagittal_Ratio'].append(r_sag)
        results['Right_Vol_Class'].append(calculate_birads(r_vol))
        results['Right_Axial_Class'].append(calculate_birads(r_ax))
        results['Right_Sagittal_Class'].append(calculate_birads(r_sag))
        
    for k, v in results.items():
        df[k] = v
    
    df.to_csv(summary_csv, index=False)
    print(f"BiRADS calculation complete. Saved to {summary_csv}")

if __name__ == "__main__":
    main()
