import os
import glob
import sys
import numpy as np
import SimpleITK as sitk
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import normalize_image, zscore_image

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config = load_config()
    registered_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference_data", "batch", "images")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(registered_dir):
        print(f"Registered dir not found: {registered_dir}")
        return
        
    subjects = [d for d in os.listdir(registered_dir) if os.path.isdir(os.path.join(registered_dir, d))]
    
    for subj in subjects:
        dyn_path = os.path.join(registered_dir, subj, f"{subj}_DYN_registered.nii.gz")
        if not os.path.exists(dyn_path):
            continue
            
        print(f"Preprocessing {subj} for FGT-Vessel inference...")
        img = sitk.ReadImage(dyn_path)
        arr = sitk.GetArrayFromImage(img)
        
        # Transform for model
        arr = np.transpose(arr, (2, 1, 0))
        arr = np.rot90(arr, k=3, axes=(0, 1))
        
        # Normalize
        arr = normalize_image(arr)
        arr = zscore_image(arr)
        
        # Pad if smaller than 96 (predict.py expects at least 96x96x96)
        pad_0 = max(0, 96 - arr.shape[0])
        pad_1 = max(0, 96 - arr.shape[1])
        pad_2 = max(0, 96 - arr.shape[2])
        if pad_0 > 0 or pad_1 > 0 or pad_2 > 0:
            arr = np.pad(arr, ((0, pad_0), (0, pad_1), (0, pad_2)), mode='constant', constant_values=np.min(arr))
            
        # Save as float32
        arr = arr.astype(np.float32)
        np.save(os.path.join(out_dir, f"{subj}.npy"), arr)

if __name__ == "__main__":
    main()
