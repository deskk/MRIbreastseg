import os
import glob
import time
import shutil
import subprocess
import SimpleITK as sitk
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="\n%(levelname)s: %(message)s")

# Hardcoded subsets limited strictly to 3 as requested.
SUBJECTS = ['046699', '076900', '081567']
DATASET_PATH = "/local/scratch/scratch-hd/desmond/dataset/clean_data_registered"
BASE_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia"
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")

PHASE_A_DIR = os.path.join(EVAL_DIR, "phase_a_combined")
PHASE_B_DIR = os.path.join(EVAL_DIR, "phase_b_separated")
NNUNET_RAW = os.path.join(BASE_DIR, "nnUNet/nnunetv2/nnUNet_raw")
PHASE_A_INPUTS = os.path.join(NNUNET_RAW, "Dataset102_Test/imagesTs")
PHASE_A_NNUNET_OUT = os.path.join(PHASE_A_DIR, "nnunet_out")

os.makedirs(PHASE_A_DIR, exist_ok=True)
os.makedirs(PHASE_A_NNUNET_OUT, exist_ok=True)
os.makedirs(PHASE_B_DIR, exist_ok=True)

def calculate_volume_ml(mask_img: sitk.Image) -> float:
    """Calculates true physical volume of the segmentation mask natively in mL."""
    spacing = mask_img.GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    arr = sitk.GetArrayViewFromImage(mask_img)
    voxel_count = np.sum(arr > 0)
    
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0
    return float(volume_ml)

def dice_score(pred, true_mask):
    intersection = np.sum(pred[true_mask == 1])
    if (np.sum(pred) + np.sum(true_mask)) == 0:
        return 1.0
    return 2.0 * intersection / (np.sum(pred) + np.sum(true_mask))

def get_reference_image(subject_dir):
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    for f in all_files:
        if "t2" not in os.path.basename(f).lower() and "sub" not in os.path.basename(f).lower():
            return f
    return all_files[0] if all_files else None

# Environment configuration
my_env = os.environ.copy()
my_env["nnUNet_raw"] = os.path.join(BASE_DIR, "nnUNet/nnunetv2/nnUNet_raw")
my_env["nnUNet_results"] = os.path.join(BASE_DIR, "nnUNet/nnunetv2/nnUNet_results")

# ==============================================================================
# PHASE A: FULL TORSO BASELINE (ACTIVE BENCHMARK)
# ==============================================================================
logging.info("--- STARTING PHASE A (Retrieving Baseline Outputs) ---")
# 1. Run nnUNet Inference on Phase A inputs dynamically
logging.info("Phase A - Native nnUNet GPU Inference Execution (Whole Torso)")

# Create an isolated input directory for just these 3 subjects to benchmark efficiently
phase_a_test_in = os.path.join(PHASE_A_DIR, "nnunet_in_subset")
os.makedirs(phase_a_test_in, exist_ok=True)

# Copy the precise 3 subject inputs for an exact benchmark array execution
for subj in SUBJECTS:
    src_file = os.path.join(PHASE_A_INPUTS, f"MAMAMIA_{subj}_0000.nii.gz")
    if os.path.exists(src_file):
        shutil.copy(src_file, os.path.join(phase_a_test_in, f"MAMAMIA_{subj}_0000.nii.gz"))

cmd_a = [
    "conda", "run", "-n", "mamamia",
    "nnUNetv2_predict", 
    "-i", phase_a_test_in, 
    "-o", PHASE_A_NNUNET_OUT, 
    "-d", "101", 
    "-c", "3d_fullres", 
    "-device", "cuda"
]

t0_a = time.time()
subprocess.run(cmd_a, env=my_env, check=True)
inference_time_a = time.time() - t0_a

# 2. Resample Phase A outputs geometric constraints
times_a = {}
volumes_a = {}

for subj in tqdm(SUBJECTS, desc="Phase A - Baseline Resampling"):
    subject_dir = os.path.join(DATASET_PATH, subj)
    pred_path = os.path.join(PHASE_A_NNUNET_OUT, f"MAMAMIA_{subj}.nii.gz")
    ref_path = get_reference_image(subject_dir)
    
    # We allocate the inference time equally across subjects for theoretical parity
    times_a[subj] = inference_time_a / len(SUBJECTS)
    
    if os.path.exists(pred_path) and ref_path:
        t_resample = time.time()
        ref_sitk = sitk.ReadImage(ref_path)
        pred_sitk = sitk.ReadImage(pred_path, sitk.sitkUInt8)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        try:
            resampled = resampler.Execute(pred_sitk)
        except:
            d = (1.,0.,0.,0.,1.,0.,0.,0.,1.)
            resampler.SetOutputDirection(d)
            pred_sitk.SetDirection(d)
            resampled = resampler.Execute(pred_sitk)
            
        out_path = os.path.join(PHASE_A_DIR, f"{subj}_baseline.nii.gz")
        sitk.WriteImage(resampled, out_path)
        
        # Calculate final mapping time
        times_a[subj] += (time.time() - t_resample)
        
        # Extract native volume metrics geometrically safely
        volumes_a[subj] = calculate_volume_ml(resampled)

# ==============================================================================
# PHASE B: SEPARATED CROP PIPELINE
# ==============================================================================
logging.info("--- STARTING PHASE B (Separated Pipeline Execution) ---")
import sys
sys.path.append(BASE_DIR)
from src.pipeline import preprocess_lr_crops
from src.pipeline import postprocess_lr_recombine

phase_b_nnunet_in = os.path.join(PHASE_B_DIR, "nnunet_in")
phase_b_nnunet_out = os.path.join(PHASE_B_DIR, "nnunet_out")
os.makedirs(phase_b_nnunet_in, exist_ok=True)
os.makedirs(phase_b_nnunet_out, exist_ok=True)

times_b = {}
volumes_b = {}

for subj in tqdm(SUBJECTS, desc="Phase B - Crop Preprocessing"):
    t0 = time.time()
    subject_dir = os.path.join(DATASET_PATH, subj)
    
    subj_out_dir = os.path.join(PHASE_B_DIR, subj)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    subj_mask_path = f"/local/scratch/scratch-hd/desmond/research/nnUNet_output/{subj}/{subj}_BreastDivider_Mask.nii.gz"
    
    preprocess_lr_crops.process_subject_sequence(subject_dir, subj_out_dir, existing_mask_path=subj_mask_path)
    
    for side in ["left", "right"]:
        crop_path = os.path.join(subj_out_dir, f"{side}_crop_1x1x1.nii.gz")
        if os.path.exists(crop_path):
            shutil.copy(crop_path, os.path.join(phase_b_nnunet_in, f"{subj}_{side}_0000.nii.gz"))
            
    times_b[subj] = time.time() - t0

logging.info("Phase B - nnUNet GPU Inference Execution")
cmd_b = [
    "conda", "run", "-n", "mamamia",
    "nnUNetv2_predict", 
    "-i", phase_b_nnunet_in, 
    "-o", phase_b_nnunet_out, 
    "-d", "101", 
    "-c", "3d_fullres", 
    "-device", "cuda"
]

t0_b = time.time()
subprocess.run(cmd_b, env=my_env, check=True)
inference_time_b = time.time() - t0_b

for s in SUBJECTS:
    times_b[s] += inference_time_b / len(SUBJECTS)

logging.info("--- RECOMBINING AND EVALUATING METRICS ---")
for subj in tqdm(SUBJECTS, desc="Phase B - Recombination & Metrics"):
    t0 = time.time()
    subject_dir = os.path.join(DATASET_PATH, subj)
    ref_path = get_reference_image(subject_dir)
    subj_out_dir = os.path.join(PHASE_B_DIR, subj)
    
    left_pred = os.path.join(phase_b_nnunet_out, f"{subj}_left.nii.gz")
    right_pred = os.path.join(phase_b_nnunet_out, f"{subj}_right.nii.gz")
    left_meta = os.path.join(subj_out_dir, "left_crop_metadata.json")
    right_meta = os.path.join(subj_out_dir, "right_crop_metadata.json")
    final_out = os.path.join(PHASE_B_DIR, f"{subj}_separated.nii.gz")
    
    if os.path.exists(left_pred) and os.path.exists(right_pred):
        postprocess_lr_recombine.process_recombination(
            original_full_torso_path=ref_path,
            left_mask_path=left_pred,
            left_metadata_path=left_meta,
            right_mask_path=right_pred,
            right_metadata_path=right_meta,
            output_combined_mask_path=final_out
        )
    
    times_b[subj] += time.time() - t0
    
    # Calculate physiological consistency variables natively
    final_img_sitk = sitk.ReadImage(final_out)
    volumes_b[subj] = calculate_volume_ml(final_img_sitk)
    
    base_mask_path = os.path.join(PHASE_A_DIR, f"{subj}_baseline.nii.gz")
    if os.path.exists(base_mask_path):
        base_arr = sitk.GetArrayFromImage(sitk.ReadImage(base_mask_path))
        sep_arr = sitk.GetArrayFromImage(final_img_sitk)
        
        base_bin = (base_arr > 0).astype(int)
        sep_bin = (sep_arr > 0).astype(int)
        
        dsc = dice_score(sep_bin, base_bin)
        
        print("\n" + "="*60)
        print(f"[{subj}] === LATENCY (End-to-End Processing Time) ===")
        print(f"    Phase A (Baseline Whole-Torso):   {times_a.get(subj, 0):.2f}s")
        print(f"    Phase B (Separated Breast Crops): {times_b.get(subj, 0):.2f}s")
        print(f"      -> Speed Optimization/Penalty: {times_a.get(subj, 0) - times_b.get(subj, 0):+.2f}s")
        
        print(f"\n[{subj}] === PHYSIOLOGICAL METRICS ===")
        print(f"    Predicted FGT/Tumor Volume (Phase A): {volumes_a.get(subj, 0):.2f} mL")
        print(f"    Predicted FGT/Tumor Volume (Phase B): {volumes_b.get(subj, 0):.2f} mL")
        print(f"      -> Topological Pipeline Agreement (Dice Score): {dsc:.4f}")
        print("="*60)
