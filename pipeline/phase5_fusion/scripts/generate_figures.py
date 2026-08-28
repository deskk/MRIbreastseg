import os
import glob
import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import json

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def find_best_slice(mask_arr):
    # Find the axial slice with the maximum FGT (3) and Tumor (5) area
    score = np.sum((mask_arr == 3) | (mask_arr == 5), axis=(1, 2))
    best_slice = np.argmax(score)
    if score[best_slice] == 0:
        # Fallback to middle slice
        best_slice = mask_arr.shape[0] // 2
    return best_slice

def main():
    config = load_config()
    fusion_dir = config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"] + "_native"
    mri_dir = config["PHASE5"]["OUTPUT_SPLIT_MRI_LEFT_DIR"]
    fig_dir = config["PHASE5"]["OUTPUT_FIGURES_DIR"]
    os.makedirs(fig_dir, exist_ok=True)
    
    subjects = [d for d in os.listdir(fusion_dir) if os.path.isdir(os.path.join(fusion_dir, d))]
    if len(subjects) < 3:
        random_subjects = subjects
    else:
        random_subjects = random.sample(subjects, 3)
        
    # Colormap: 0:Bg, 1:Fat, 2:Skin, 3:FGT, 4:Vessels, 5:Tumor
    colors = [
        (0,0,0,0),            # 0: Transparent background
        (1, 1, 0, 0.4),       # 1: Fat (Yellow)
        (0, 1, 1, 0.5),       # 2: Skin (Cyan)
        (0, 1, 0, 0.6),       # 3: FGT (Green)
        (1, 0, 0, 0.8),       # 4: Vessels (Red)
        (1, 0, 1, 0.8)        # 5: Tumor (Magenta)
    ]
    cmap = ListedColormap(colors)
    
    # Legend patches
    legend_patches = [
        mpatches.Patch(color='yellow', label='Fat', alpha=0.4),
        mpatches.Patch(color='cyan', label='Skin', alpha=0.5),
        mpatches.Patch(color='green', label='FGT', alpha=0.6),
        mpatches.Patch(color='red', label='Vessels', alpha=0.8),
        mpatches.Patch(color='magenta', label='Tumor', alpha=0.8)
    ]

    for subj in random_subjects:
        mri_path = os.path.join(mri_dir, subj, f"{subj}_DYN_cropped.nii.gz")
        fusion_path = os.path.join(fusion_dir, subj, f"{subj}_final_fusion.nii.gz")
        
        if not os.path.exists(mri_path) or not os.path.exists(fusion_path):
            print(f"Files missing for {subj}, skipping.")
            continue
            
        mri_img = sitk.ReadImage(mri_path)
        mask_img = sitk.ReadImage(fusion_path)
        
        mri_arr = sitk.GetArrayFromImage(mri_img)
        mask_arr = sitk.GetArrayFromImage(mask_img)
        
        z = find_best_slice(mask_arr)
        
        mri_slice = mri_arr[z, :, :]
        mask_slice = mask_arr[z, :, :]
        
        # Stage 1: MRI only
        stage1_img = mri_slice
        
        # Stage 2: Internal tissues (FGT=3, Vessels=4, Tumor=5)
        stage2_mask = np.zeros_like(mask_slice)
        stage2_mask[(mask_slice >= 3) & (mask_slice <= 5)] = mask_slice[(mask_slice >= 3) & (mask_slice <= 5)]
        
        # Stage 3: Full envelope (Everything)
        stage3_mask = mask_slice
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        plt.subplots_adjust(wspace=0.01, hspace=0)
        
        vmax = np.percentile(mri_slice, 99)
        
        for ax in axes:
            ax.axis('off')
            
        # Stage 1 Plot
        axes[0].imshow(mri_slice, cmap='gray', vmax=vmax)
        axes[0].set_title('Stage 1: Original Image', color='white', pad=-20, y=1.0)
        
        # Stage 2 Plot
        axes[1].imshow(mri_slice, cmap='gray', vmax=vmax)
        axes[1].imshow(stage2_mask, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
        axes[1].set_title('Stage 2: Internal Tissues', color='white', pad=-20, y=1.0)
        
        # Stage 3 Plot
        axes[2].imshow(mri_slice, cmap='gray', vmax=vmax)
        axes[2].imshow(stage3_mask, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
        axes[2].set_title('Stage 3: Full Model (+Skin/Fat)', color='white', pad=-20, y=1.0)
        
        # Set dark background figure
        fig.patch.set_facecolor('black')
        
        # Add legend
        fig.legend(handles=legend_patches, loc='lower center', ncol=5, frameon=False, labelcolor='white', bbox_to_anchor=(0.5, 0.05))
        
        out_path = os.path.join(fig_dir, f"{subj}_miccai_fig.png")
        plt.savefig(out_path, bbox_inches='tight', facecolor='black', pad_inches=0.1)
        plt.close(fig)
        
        print(f"Generated {out_path}")

if __name__ == '__main__':
    main()
