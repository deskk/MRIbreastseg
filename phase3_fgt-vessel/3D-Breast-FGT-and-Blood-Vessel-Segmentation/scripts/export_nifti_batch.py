import os
import glob
import numpy as np
import SimpleITK as sitk
import json

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def inverse_transform(arr):
    # Reverse of np.rot90(arr, k=3, axes=(0, 1))
    arr = np.rot90(arr, k=1, axes=(0, 1))
    
    # Reverse of np.transpose(arr, (2, 1, 0)) -> This is its own inverse!
    arr = np.transpose(arr, (2, 1, 0))
    return arr

def main():
    config = load_config()
    registered_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]
    out_fgt_vessel = config["PHASE3"]["OUTPUT_FGT_VESSEL_DIR"]
    
    preds_dv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference_data", "batch", "preds_dv")
    
    if not os.path.exists(preds_dv_dir):
        print(f"Predictions dir not found: {preds_dv_dir}")
        return
        
    os.makedirs(out_fgt_vessel, exist_ok=True)
    
    pred_files = glob.glob(os.path.join(preds_dv_dir, "*.npy"))
    
    for pred_file in pred_files:
        subj = os.path.basename(pred_file).replace('.npy', '')
        print(f"Exporting NIfTI for {subj}...")
        
        dyn_path = os.path.join(registered_dir, subj, f"{subj}_DYN_registered.nii.gz")
        if not os.path.exists(dyn_path):
            print(f"  Missing reference image {dyn_path}")
            continue
            
        ref_img = sitk.ReadImage(dyn_path)
        
        arr = np.load(pred_file)
        # Prediction might be (X, Y, Z, 3) where channels are background, vessel, fgt?
        # Actually usually argmax is already applied, so it's (X, Y, Z) with labels 0, 1, 2.
        if len(arr.shape) == 4:
            # Classes are on the first axis (n_classes, x, y, z)
            arr = np.argmax(arr, axis=0)
            
        arr = inverse_transform(arr)
        
        orig_shape = sitk.GetArrayFromImage(ref_img).shape
        arr = arr[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
        
        arr = arr.astype(np.uint8)
        
        out_img = sitk.GetImageFromArray(arr)
        out_img.CopyInformation(ref_img)
        
        subj_out_dir = os.path.join(out_fgt_vessel, subj)
        os.makedirs(subj_out_dir, exist_ok=True)
        sitk.WriteImage(out_img, os.path.join(subj_out_dir, f"{subj}_dv_mask.nii.gz"))

if __name__ == "__main__":
    main()
