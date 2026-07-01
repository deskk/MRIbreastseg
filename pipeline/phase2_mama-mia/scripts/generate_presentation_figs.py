import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia"
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
DATASET_PATH = "/local/scratch/scratch-hd/desmond/dataset/clean_data_registered"
PHASE_A_DIR = os.path.join(EVAL_DIR, "phase_a_combined")
PHASE_B_DIR = os.path.join(EVAL_DIR, "phase_b_separated")

FIG_OUT_DIR = os.path.join(BASE_DIR, "presentation_figures")
os.makedirs(FIG_OUT_DIR, exist_ok=True)

SUBJECTS = ['046699', '076900', '081567']

def get_reference_image(subject_dir):
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    for f in all_files:
        if "t2" not in os.path.basename(f).lower() and "sub" not in os.path.basename(f).lower():
            return f
    return all_files[0] if all_files else None

# 1. GENERATE FALSE POSITIVE BAR CHART
print("Generating Quantitative Bar Chart...")
fp_counts = [481, 638, 284]  # We extracted these from the exact log outputs earlier natively
subjects = ['Subj 046699', 'Subj 076900', 'Subj 081567']

plt.figure(figsize=(8, 6))
bars = plt.bar(subjects, fp_counts, color=['#e63946', '#e63946', '#e63946'])
plt.title("MAMA-MIA Baseline Hallucinations (False Positives Suppressed)", fontsize=14, fontweight='bold')
plt.ylabel("Extraneous Voxels (Out-of-Bounds)", fontsize=12)
plt.xlabel("Test Subjects", fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
chart_path = os.path.join(FIG_OUT_DIR, "false_positives_barchart.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved Bar Chart -> {chart_path}\n")

# 2. GENERATE ANATOMICAL OVERLAYS
print("Generating Anatomical MRI Overlays for visual presentation...")
# Custom colormaps
cmap_red = ListedColormap(['none', 'red'])
cmap_green = ListedColormap(['none', 'lime'])

for subj in SUBJECTS:
    base_mask_path = os.path.join(PHASE_A_DIR, f"{subj}_baseline.nii.gz")
    sep_mask_path = os.path.join(PHASE_B_DIR, f"{subj}_separated.nii.gz")
    ref_path = get_reference_image(os.path.join(DATASET_PATH, subj))
    
    if not (os.path.exists(base_mask_path) and os.path.exists(sep_mask_path) and ref_path):
        continue
        
    print(f"Processing anatomical slices for Subject: {subj}")
    ref_arr = sitk.GetArrayFromImage(sitk.ReadImage(ref_path)).astype(np.float32)
    base_arr = sitk.GetArrayFromImage(sitk.ReadImage(base_mask_path))
    sep_arr = sitk.GetArrayFromImage(sitk.ReadImage(sep_mask_path))
    
    # Binarize masks
    base_bin = (base_arr > 0).astype(int)
    sep_bin = (sep_arr > 0).astype(int)
    
    # Isolate False Positives (Baseline thought it was tumor/FGT, but Separated correctly ignored it)
    false_positives = (base_bin == 1) & (sep_bin == 0)
    
    # Find the axial Z-slice with the maximum absolute number of False Positive voxels
    z_fp_counts = np.sum(false_positives, axis=(1, 2))
    
    if np.sum(z_fp_counts) == 0:
        # If no false positives, just pick the slice with the largest tumor/FGT volume in separated mask
        z_slice = np.argmax(np.sum(sep_bin, axis=(1, 2)))
        print(f"    -> No FP found, defaulting to largest True Positive slice Z={z_slice}")
    else:
        z_slice = np.argmax(z_fp_counts)
        print(f"    -> Found peak hallucination slice at Z={z_slice} with {z_fp_counts[z_slice]} aberrant voxels")
        
    base_overlay = base_bin[z_slice]
    sep_overlay = sep_bin[z_slice]
    bg_slice = ref_arr[z_slice]
    
    # Normalize background slice for proper grayscale viewing contrast
    bg_slice = np.clip(bg_slice, np.percentile(bg_slice, 2), np.percentile(bg_slice, 98))
    bg_slice = (bg_slice - np.min(bg_slice)) / (np.max(bg_slice) - np.min(bg_slice) + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Subject {subj} Axial Slice Z={z_slice} Comparison", fontsize=16, fontweight='bold')
    
    # Left subplot: Baseline (Red)
    axes[0].imshow(bg_slice, cmap='gray')
    axes[0].imshow(np.ma.masked_where(base_overlay == 0, base_overlay), cmap=cmap_red, alpha=0.6, vmin=0, vmax=1)
    axes[0].set_title("Phase A: Whole-Torso Baseline\n(Red = Hallucinated/Mask)", fontsize=14)
    axes[0].axis('off')
    
    # Right subplot: Separated (Green)
    axes[1].imshow(bg_slice, cmap='gray')
    axes[1].imshow(np.ma.masked_where(sep_overlay == 0, sep_overlay), cmap=cmap_green, alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title("Phase B: BreastDivider Separated\n(Green = Mathematically Bounded)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    comp_path = os.path.join(FIG_OUT_DIR, f"{subj}_comparison_overlay.png")
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved visual overlay -> {comp_path}")

print("\nPresentation Figures generated successfully.")
