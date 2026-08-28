import os
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

# UU Dataset Paths
UU_MRI_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/data-uu/phase1_left"
UU_MASK_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/data-uu/phase5_fusion/left"

# Duke Dataset Paths
DUKE_MRI_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external-duke-fgt/duke_outputs/phase5_split_mri/left"
DUKE_MASK_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external-duke-fgt/duke_outputs/phase5_fusion/left_native"

OUT_DIR = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/paper_fig"
os.makedirs(OUT_DIR, exist_ok=True)

def get_most_representative_slice(mask_data):
    # Sum over X, Y to get the area per Z slice
    slice_areas = np.sum(mask_data > 0, axis=(0, 1))
    best_slice_idx = np.argmax(slice_areas)
    if slice_areas[best_slice_idx] == 0:
        return mask_data.shape[2] // 2
    return best_slice_idx

def get_perc_clip(mri_data, p=99):
    return np.percentile(mri_data, p)

def plot_and_save(mri_data, mask_data, slice_idx, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    
    mri_slice = mri_data[:, :, slice_idx].T
    mask_slice = mask_data[:, :, slice_idx].T
    
    mri_slice = np.flipud(mri_slice)
    mask_slice = np.flipud(mask_slice)

    vmax = get_perc_clip(mri_data)
    
    # Plot MRI
    axes[0].imshow(mri_slice, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title(f"(a) MR Image", fontsize=14, pad=10)
    axes[0].axis('off')
    
    # Plot MRI + Mask
    axes[1].imshow(mri_slice, cmap='gray', vmin=0, vmax=vmax)
    masked_data = np.ma.masked_where(mask_slice == 0, mask_slice)
    # Using a categorical map for fusion
    # classes: 1:Fat, 2:Skin, 3:FGT, 4:Vessels, 5:Tumor
    # To keep it simple and high contrast:
    cmap = plt.get_cmap('Set1', 6)
    axes[1].imshow(masked_data, cmap=cmap, alpha=0.5, vmin=0, vmax=5)
    axes[1].set_title(f"(b) Final Fusion Mask", fontsize=14, pad=10)
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)

def process_uu():
    uu_patients = os.listdir(UU_MASK_DIR)
    uu_patients = [p for p in uu_patients if os.path.isdir(os.path.join(UU_MASK_DIR, p))]
    random.seed(42)
    random.shuffle(uu_patients)
    
    count = 0
    for pt in uu_patients:
        if count >= 20: break
        
        mask_path = os.path.join(UU_MASK_DIR, pt, f"{pt}_final_fusion.nii.gz")
        if not os.path.exists(mask_path):
            continue
            
        mri_dir = os.path.join(UU_MRI_DIR, pt)
        mri_files = glob(os.path.join(mri_dir, "*T1*.nii.gz"))
        if not mri_files: continue
        mri_path = sorted(mri_files)[0]
        
        try:
            mri_img = nib.load(mri_path)
            mask_img = nib.load(mask_path)
            
            mri_data = mri_img.get_fdata()
            mask_data = mask_img.get_fdata()
            
            if mri_data.shape != mask_data.shape:
                print(f"Skipping {pt} due to shape mismatch: {mri_data.shape} != {mask_data.shape}")
                continue
                
            best_slice = get_most_representative_slice(mask_data)
            
            out_file = os.path.join(OUT_DIR, f"UU_{pt}_slice{best_slice}.png")
            plot_and_save(mri_data, mask_data, best_slice, out_file, f"UU Dataset - Patient {pt}")
            count += 1
            print(f"Generated UU {count}/20")
        except Exception as e:
            print(f"Error UU pt {pt}: {e}")
            
def process_duke():
    duke_patients = os.listdir(DUKE_MASK_DIR)
    duke_patients = [p for p in duke_patients if os.path.isdir(os.path.join(DUKE_MASK_DIR, p))]
    random.seed(42)
    random.shuffle(duke_patients)
    
    count = 0
    for pt in duke_patients:
        if count >= 20: break
        
        mask_path = os.path.join(DUKE_MASK_DIR, pt, f"{pt}_final_fusion.nii.gz")
        if not os.path.exists(mask_path):
            continue
            
        mri_path = os.path.join(DUKE_MRI_DIR, pt, f"{pt}_DYN_cropped.nii.gz")
        if not os.path.exists(mri_path):
            # Check other combinations if DYN_cropped is missing
            mri_files = glob(os.path.join(DUKE_MRI_DIR, pt, "*.nii.gz"))
            if mri_files:
                mri_path = mri_files[0]
            else:
                continue
        
        try:
            mri_img = nib.load(mri_path)
            mask_img = nib.load(mask_path)
            
            mri_data = mri_img.get_fdata()
            mask_data = mask_img.get_fdata()
            
            if mri_data.shape != mask_data.shape:
                print(f"Skipping Duke {pt} due to shape mismatch: {mri_data.shape} != {mask_data.shape}")
                continue
                
            best_slice = get_most_representative_slice(mask_data)
            
            out_file = os.path.join(OUT_DIR, f"Duke_{pt}_slice{best_slice}.png")
            plot_and_save(mri_data, mask_data, best_slice, out_file, f"Duke Dataset - {pt}")
            count += 1
            print(f"Generated Duke {count}/20")
        except Exception as e:
            print(f"Error Duke pt {pt}: {e}")

if __name__ == "__main__":
    process_uu()
    process_duke()
    print("Done")
