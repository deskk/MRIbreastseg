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
    if fgt_ratio < 0.25:
        return 'A'
    elif fgt_ratio < 0.50:
        return 'B'
    elif fgt_ratio < 0.75:
        return 'C'
    else:
        return 'D'

def get_fgt_ratio(fusion_path):
    if not os.path.exists(fusion_path):
        return None
    
    img = sitk.ReadImage(fusion_path)
    arr = sitk.GetArrayFromImage(img)
    
    # 1 is Fat, 3 is FGT
    fat_pixels = np.sum(arr == 1)
    fgt_pixels = np.sum(arr == 3)
    
    total = fat_pixels + fgt_pixels
    if total == 0:
        return 0.0
    
    return float(fgt_pixels) / float(total)

def main():
    config = load_config()
    left_dir = config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"]
    right_dir = config["PHASE5"]["OUTPUT_FUSION_RIGHT_DIR"]
    
    in_csv = config["PHASE5"]["OUTPUT_TUMOR_PRESENCE_CSV"]
    out_csv = config["PHASE5"]["OUTPUT_TUMOR_BIRADS_CSV"]
    
    if not os.path.exists(in_csv):
        print("Missing tumor presence CSV!")
        return
        
    df = pd.read_csv(in_csv)
    
    left_ratios = []
    right_ratios = []
    left_birads = []
    right_birads = []
    
    for _, row in df.iterrows():
        subj = row['Subject']
        
        left_path = os.path.join(left_dir, subj, f"{subj}_final_fusion.nii.gz")
        right_path = os.path.join(right_dir, subj, f"{subj}_final_fusion.nii.gz")
        
        l_ratio = get_fgt_ratio(left_path)
        r_ratio = get_fgt_ratio(right_path)
        
        left_ratios.append(l_ratio)
        right_ratios.append(r_ratio)
        left_birads.append(calculate_birads(l_ratio) if l_ratio is not None else None)
        right_birads.append(calculate_birads(r_ratio) if r_ratio is not None else None)
        
    df['Left_FGT_Ratio'] = left_ratios
    df['Left_BiRADS'] = left_birads
    df['Right_FGT_Ratio'] = right_ratios
    df['Right_BiRADS'] = right_birads
    
    df.to_csv(out_csv, index=False)
    print(f"BiRADS calculation complete. Saved to {out_csv}")

if __name__ == "__main__":
    main()
