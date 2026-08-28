import os
import glob
import logging
import tempfile
import subprocess
import shutil
import numpy as np
import SimpleITK as sitk
import json

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

INPUT_DIR = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]
OUTPUT_DIR = config["PHASE2"]["OUTPUT_FULL_DIR"]

TARGET_SPACING = (1.0, 1.0, 1.0)

def get_dce_sequences(subject_dir):
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    pre_file = None
    post_files = []
    
    for f in all_files:
        filename = os.path.basename(f).upper()
        if "SUB" in filename or "MASK" in filename or "T2" in filename or ("T1" in filename and "DYNAMIC" not in filename and "DYN" not in filename and "POST" not in filename):
            continue
            
        if "PRE" in filename:
            pre_file = f
        elif "DYNAMIC" in filename or "DYN" in filename or "POST" in filename:
            post_files.append(f)
            
    post_files.sort()
    return pre_file, post_files

def compute_global_statistics(all_dce_files):
    all_pixels = []
    for f in all_dce_files:
        img_sitk = sitk.ReadImage(f, sitk.sitkFloat32)
        arr = sitk.GetArrayViewFromImage(img_sitk).ravel()
        arr = arr[arr != 0]
        if len(arr) > 0:
            all_pixels.append(arr)
        
    if not all_pixels:
        return 0.0, 1.0
        
    all_pixels = np.concatenate(all_pixels)
    global_mean = float(np.mean(all_pixels))
    global_std = float(np.std(all_pixels))
    
    logging.info(f"Global Normalization Statistics -> Mean: {global_mean:.4f}, Std: {global_std:.4f}")
    return global_mean, global_std

def normalize_and_resample(image_path, global_mean, global_std):
    img = sitk.ReadImage(image_path, sitk.sitkFloat32)
    
    # Normalize
    normalized_img = sitk.ShiftScale(img, shift=-global_mean, scale=1.0 / global_std)
    
    # Resample
    original_spacing = normalized_img.GetSpacing()
    original_size = normalized_img.GetSize()
    
    resampled_size = [
        int(np.round(sz * (spc / tgt_spc)))
        for sz, spc, tgt_spc in zip(original_size, original_spacing, TARGET_SPACING)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(resampled_size)
    resampler.SetOutputOrigin(normalized_img.GetOrigin())
    resampler.SetOutputDirection(normalized_img.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    
    final_img = resampler.Execute(normalized_img)
    return final_img

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subjects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    test_subjects = config.get("TEST_SUBJECTS", [])
    if test_subjects:
        subjects = [s for s in subjects if s in test_subjects]
    
    with tempfile.TemporaryDirectory() as temp_in, tempfile.TemporaryDirectory() as temp_out:
        valid_subjects = []
        
        for subj in subjects:
            expected_mask = os.path.join(output_dir, subj, f"{subj}_MAMAMIA_Mask.nii.gz")
            if os.path.exists(expected_mask):
                logging.info(f"Skipping {subj}, Phase 2 mask already exists.")
                continue

            subj_dir = os.path.join(input_dir, subj)
            pre_file, post_files = get_dce_sequences(subj_dir)
            
            if not post_files:
                logging.error(f"Missing POST sequences for {subj} in {input_dir}. Skipping.")
                continue
                
            all_dce = post_files.copy()
            if pre_file:
                all_dce.append(pre_file)
                
            logging.info(f"Processing {subj}...")
            mean, std = compute_global_statistics(all_dce)
            
            first_post = post_files[0]
            preprocessed_img = normalize_and_resample(first_post, mean, std)
            
            in_file = os.path.join(temp_in, f"{subj}_0000.nii.gz")
            sitk.WriteImage(preprocessed_img, in_file)
            
            final_subj_dir = os.path.join(output_dir, subj)
            os.makedirs(final_subj_dir, exist_ok=True)
            final_in_path = os.path.join(final_subj_dir, f"{subj}_MAMAMIA_Input_1stPOST.nii.gz")
            sitk.WriteImage(preprocessed_img, final_in_path)
            logging.info(f"Saved Input MRI: {final_in_path}")
            
            valid_subjects.append(subj)
            
        if not valid_subjects:
            logging.error(f"No valid subjects prepared in {input_dir}. Aborting inference.")
            return
            
        logging.info(f"Running Phase 2 Inference on {len(valid_subjects)} subjects from {input_dir}...")
        
        cmd = [
            "nnUNetv2_predict",
            "-i", temp_in,
            "-o", temp_out,
            "-d", "101",
            "-c", "3d_fullres"
        ]
        
        env = os.environ.copy()
        env["nnUNet_raw"] = config["PHASE2"]["NNUNET_RAW"]
        env["nnUNet_preprocessed"] = config["PHASE2"]["NNUNET_PREPROCESSED"]
        env["nnUNet_results"] = config["PHASE2"]["NNUNET_RESULTS"]
        
        subprocess.run(cmd, env=env, check=True)
        
        for subj in valid_subjects:
            out_file = os.path.join(temp_out, f"{subj}.nii.gz")
            if os.path.exists(out_file):
                pred_mask = sitk.ReadImage(out_file, sitk.sitkUInt8)
                
                subj_dir = os.path.join(input_dir, subj)
                _, post_files = get_dce_sequences(subj_dir)
                ref_img = sitk.ReadImage(post_files[0])
                
                resampler = sitk.ResampleImageFilter()
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetOutputSpacing(ref_img.GetSpacing())
                resampler.SetSize(ref_img.GetSize())
                resampler.SetOutputDirection(ref_img.GetDirection())
                resampler.SetOutputOrigin(ref_img.GetOrigin())
                resampler.SetTransform(sitk.Transform())
                resampler.SetDefaultPixelValue(0)
                
                final_mask = resampler.Execute(pred_mask)
                
                final_subj_dir = os.path.join(output_dir, subj)
                os.makedirs(final_subj_dir, exist_ok=True)
                final_path = os.path.join(final_subj_dir, f"{subj}_MAMAMIA_Mask.nii.gz")
                sitk.WriteImage(final_mask, final_path)
                logging.info(f"Saved {final_path}")
            else:
                logging.error(f"Failed to generate output for {subj}")

def main():
    if not os.path.exists(INPUT_DIR):
        logging.warning(f"Input directory does not exist: {INPUT_DIR}")
        return
        
    logging.info(f"\n=== Processing Directory: {INPUT_DIR} ===")
    process_directory(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
