import os
import glob
import random
import shutil
import tempfile
import subprocess
import json

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config()

INPUT_DIR = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]
OUTPUT_DIR = config["PHASE1"]["OUTPUT_MASK_DIR"]
MODEL_DIR = config["PHASE1"]["MODEL_DIR"]

def get_pre_contrast_series(subject_dir):
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    for f in all_files:
        filename = os.path.basename(f).lower()
        # DIFFERENCE REPORTED: Original script required both "pre" and "t1" in the filename. 
        # For generalization across datasets (like DUKE) and standardized Phase 0 outputs, 
        # we check for "pre" or "t1" independently.
        if "pre" in filename or "t1" in filename:
            return f
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print("Input directory does not exist.")
        return
    
    # Get all subjects
    subjects = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    test_subjects = config.get("TEST_SUBJECTS", [])
    if test_subjects:
        subjects = [s for s in subjects if s in test_subjects]
        
    print(f"Total subjects found in registered: {len(subjects)}")
    
    with tempfile.TemporaryDirectory() as temp_in, tempfile.TemporaryDirectory() as temp_out:
        # Prepare inputs
        valid_subjects = []
        for subj in subjects:
            # SKIP LOGIC
            expected_output = os.path.join(OUTPUT_DIR, subj, f"{subj}_BreastDivider_Mask.nii.gz")
            if os.path.exists(expected_output):
                continue
                
            subj_dir = os.path.join(INPUT_DIR, subj)
            img_path = get_pre_contrast_series(subj_dir)
            if img_path:
                in_file = os.path.join(temp_in, f"{subj}_0000.nii.gz")
                shutil.copy(img_path, in_file)
                valid_subjects.append(subj)
            else:
                print(f"Warning: No T1 PRE image found for subject {subj}")
                
        print(f"Running Phase 1 inference on {len(valid_subjects)} remaining valid subjects...")
        
        if not valid_subjects:
            print("No new subjects to process. Exiting.")
            return
        
        cmd = [
            "nnUNetv2_predict_from_modelfolder",
            "-i", temp_in,
            "-o", temp_out,
            "-m", MODEL_DIR,
            "-f", "all"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Move outputs to final destination
        for subj in valid_subjects:
            out_file = os.path.join(temp_out, f"{subj}.nii.gz")
            if os.path.exists(out_file):
                final_subj_dir = os.path.join(OUTPUT_DIR, subj)
                os.makedirs(final_subj_dir, exist_ok=True)
                final_path = os.path.join(final_subj_dir, f"{subj}_BreastDivider_Mask.nii.gz")
                shutil.copy(out_file, final_path)
                print(f"Saved {final_path}")
            else:
                print(f"Error: Output not generated for {subj}")

if __name__ == "__main__":
    main()
