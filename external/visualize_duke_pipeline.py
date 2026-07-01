import os
import json
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def get_best_slice(mri_arr):
    # Just grab the middle slice for consistency
    return mri_arr.shape[0] // 2

def safe_read_array(path):
    if not os.path.exists(path):
        return None
    try:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    except:
        return None

def main():
    import random
    config = load_config()
    test_subjects = config.get("TEST_SUBJECTS", [])
    if not test_subjects:
        all_subjects = os.listdir(config["PHASE0"]["REGISTERED_OUTPUT_DIR"])
        all_subjects = [s for s in all_subjects if s.startswith("Breast_MRI_")]
        
        if len(all_subjects) == 0:
            print("No subjects found.")
            return
            
        random.shuffle(all_subjects)
        test_subjects = all_subjects[:10]

    out_figs_dir = config["PHASE5"]["OUTPUT_FIGURES_DIR"]
    os.makedirs(out_figs_dir, exist_ok=True)

    for subj in test_subjects:
        print(f"Generating visual report for {subj}...")
        
        # Load arrays
        mri_path = os.path.join(config["PHASE0"]["REGISTERED_OUTPUT_DIR"], subj, f"{subj}_DYN1_registered.nii.gz")
        if not os.path.exists(mri_path):
            mri_path = os.path.join(config["PHASE0"]["REGISTERED_OUTPUT_DIR"], subj, f"{subj}_PRE_registered.nii.gz")
        
        mri_arr = safe_read_array(mri_path)
        if mri_arr is None:
            print(f"Skipping {subj}, no MRI found.")
            continue
            
        bd_arr = safe_read_array(os.path.join(config["PHASE1"]["OUTPUT_MASK_DIR"], subj, f"{subj}_BreastDivider_Mask.nii.gz"))
        tumor_arr = safe_read_array(os.path.join(config["PHASE2"]["OUTPUT_FULL_DIR"], subj, f"{subj}_MAMAMIA_Mask.nii.gz"))
        dv_arr = safe_read_array(os.path.join(config["PHASE3"]["OUTPUT_FGT_VESSEL_DIR"], subj, f"{subj}_dv_mask.nii.gz"))
        fat_arr = safe_read_array(os.path.join(config["PHASE4"]["OUTPUT_FAT_DIR"], subj, f"{subj}_fat_mask.nii.gz"))
        skin_arr = safe_read_array(os.path.join(config["PHASE4"]["OUTPUT_SKIN_DIR"], subj, f"{subj}_skin_mask.nii.gz"))
        fusion_left_arr = safe_read_array(os.path.join(config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"], subj, f"{subj}_final_fusion.nii.gz"))

        z = get_best_slice(mri_arr)
        mri_slice = mri_arr[z]
        vmax = np.percentile(mri_slice, 99)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Pipeline Progress for {subj} (Slice {z})', fontsize=20, color='black')
        fig.patch.set_facecolor('white')
        axes = axes.flatten()

        for ax in axes:
            ax.set_facecolor('white')
            ax.axis('off')
            ax.imshow(mri_slice, cmap='gray', vmax=vmax)

        # 1. Phase 0: Raw MRI
        axes[0].set_title("Phase 0: DYN1 Registered", color='black')

        # 2. Phase 1: BreastDivider
        axes[1].set_title("Phase 1: Torso Segmentation", color='black')
        if bd_arr is not None:
            bd_slice = bd_arr[z]
            axes[1].imshow(np.ma.masked_where(bd_slice == 0, bd_slice), cmap='cool', alpha=0.5, interpolation='nearest')

        # 3. Phase 2: Tumor
        axes[2].set_title("Phase 2: Tumor Mask", color='black')
        if tumor_arr is not None:
            tumor_slice = tumor_arr[z]
            axes[2].imshow(np.ma.masked_where(tumor_slice == 0, tumor_slice), cmap='spring', alpha=0.8, interpolation='nearest')

        # 4. Phase 3: FGT/Vessels
        axes[3].set_title("Phase 3: FGT (Blue) & Vessels (Red)", color='black')
        if dv_arr is not None:
            dv_slice = dv_arr[z]
            dv_cmap = ListedColormap([(0,0,0,0), (1,0,0,0.6), (0,0,1,0.6)])
            axes[3].imshow(dv_slice, cmap=dv_cmap, vmin=0, vmax=2, interpolation='nearest')

        # 5. Phase 4: Skin/Fat
        axes[4].set_title("Phase 4: Skin (Cyan) & Fat (Yellow)", color='black')
        if fat_arr is not None and skin_arr is not None:
            combo = np.zeros_like(fat_arr[z])
            combo[fat_arr[z] == 1] = 1
            combo[skin_arr[z] == 1] = 2
            sf_cmap = ListedColormap([(0,0,0,0), (1,1,0,0.3), (0,1,1,0.6)])
            axes[4].imshow(combo, cmap=sf_cmap, vmin=0, vmax=2, interpolation='nearest')

        # 6. Phase 5: Fusion Left
        axes[5].clear()
        axes[5].set_facecolor('white')
        axes[5].axis('off')
        axes[5].set_title("Phase 5: Final Fusion (Left Breast)", color='black')
        
        mri_left_arr = safe_read_array(os.path.join(config["PHASE5"]["OUTPUT_SPLIT_MRI_LEFT_DIR"], subj, f"{subj}_DYN_cropped.nii.gz"))
        if mri_left_arr is not None and fusion_left_arr is not None:
            fused_z = fusion_left_arr.shape[0] // 2
            axes[5].imshow(mri_left_arr[fused_z], cmap='gray', vmax=np.percentile(mri_left_arr[fused_z], 99))
            axes[5].imshow(np.ma.masked_where(fusion_left_arr[fused_z] == 0, fusion_left_arr[fused_z]), cmap='jet', alpha=0.6, interpolation='nearest')

        plt.tight_layout()
        out_file = os.path.join(out_figs_dir, f"{subj}_pipeline_visual.png")
        plt.savefig(out_file, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Generated {out_file}")

if __name__ == '__main__':
    main()
