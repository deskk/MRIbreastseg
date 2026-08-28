import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import shutil

def process_subject(subj, cls, out_dir, artifact_dir):
    subj_str = str(subj).zfill(6)
    base_dir = '/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/data-uu/archive'
    side = 'left'
    
    mri_path = os.path.join(base_dir, 'phase2_output', side, subj_str, f"{subj_str}_MAMAMIA_Input_1stPOST.nii.gz")
    fusion_path = os.path.join(base_dir, 'phase5_fusion', side, subj_str, f"{subj_str}_final_fusion.nii.gz")
    
    if not os.path.exists(mri_path) or not os.path.exists(fusion_path):
        return False
        
    mri_img = sitk.ReadImage(mri_path)
    fusion_img = sitk.ReadImage(fusion_path)
    
    # Resample MRI to match Fusion image space
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fusion_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    mri_img_resampled = resampler.Execute(mri_img)
    
    mri_arr = sitk.GetArrayFromImage(mri_img_resampled)
    fusion_arr = sitk.GetArrayFromImage(fusion_img)
    
    breast_mask = (fusion_arr == 1) | (fusion_arr == 3)
    fgt_mask = (fusion_arr == 3)
    
    masked_mri = mri_arr.copy()
    masked_mri[~breast_mask] = 0
    
    # Sagittal MIP
    sag_mip_mri = np.max(masked_mri, axis=2)
    sag_mip_fgt = np.any(fgt_mask, axis=2)
    sag_mip_breast = np.any(breast_mask, axis=2)
    
    # Crop to breast bounding box
    rows = np.any(sag_mip_breast, axis=1)
    cols = np.any(sag_mip_breast, axis=0)
    
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        pad = 20
        ymin = max(0, ymin - pad)
        ymax = min(sag_mip_mri.shape[0], ymax + pad)
        xmin = max(0, xmin - pad)
        xmax = min(sag_mip_mri.shape[1], xmax + pad)
        
        sag_mip_mri = sag_mip_mri[ymin:ymax, xmin:xmax]
        sag_mip_fgt = sag_mip_fgt[ymin:ymax, xmin:xmax]
    
    # Ratios
    sag_fgt_area = np.sum(sag_mip_fgt)
    sag_breast_area = np.sum(np.any(breast_mask, axis=2))
    sag_ratio = float(sag_fgt_area) / float(sag_breast_area) if sag_breast_area > 0 else 0
    
    z_slices = fusion_arr.shape[0]
    slice_ratios = []
    for z in range(z_slices):
        fat = np.sum(fusion_arr[z] == 1)
        fgt = np.sum(fusion_arr[z] == 3)
        total = fat + fgt
        slice_ratios.append(float(fgt) / float(total) if total > 0 else 0.0)
        
    total_fat = np.sum(fusion_arr == 1)
    total_fgt = np.sum(fusion_arr == 3)
    overall_ratio = float(total_fgt) / float(total_fat + total_fgt) if (total_fat + total_fgt) > 0 else 0
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    # Plot Sagittal MIP (rotated to be upright)
    axes[0].imshow(np.rot90(sag_mip_mri), cmap='gray')
    overlay_sag = np.zeros((*sag_mip_fgt.shape, 4))
    overlay_sag[sag_mip_fgt] = [1, 0, 0, 0.4] 
    axes[0].imshow(np.rot90(overlay_sag))
    axes[0].set_title(f"Sagittal MIP (FGT: {sag_ratio:.1%})")
    axes[0].axis('off')
    
    # Plot Distribution
    axes[1].plot(range(z_slices), slice_ratios, color='#1f77b4', linewidth=1.5, label='Per-Slice (3D)')
    axes[1].axhline(y=overall_ratio, color='red', linestyle='--', linewidth=1.5, label=f'3D Volumetric Ratio ({overall_ratio:.1%})')
    axes[1].axhline(y=sag_ratio, color='purple', linestyle=':', linewidth=1.5, label=f'2D Sagittal MIP ({sag_ratio:.1%})')
    
    axes[1].axhline(y=0.25, color='gray', linestyle=':', linewidth=0.5)
    axes[1].axhline(y=0.50, color='gray', linestyle=':', linewidth=0.5)
    axes[1].axhline(y=0.75, color='gray', linestyle=':', linewidth=0.5)
    
    axes[1].set_xlabel('Z-Slice Index')
    axes[1].set_ylabel('FGT Ratio')
    axes[1].set_title('FGT Distribution Comparison')
    axes[1].legend(fontsize=9, loc='upper right', frameon=False)
    axes[1].set_ylim(0, 1.0)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f'sagittal_example_class_{cls}_{subj_str}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    shutil.copy2(out_path, os.path.join(artifact_dir, f'sagittal_example_class_{cls}_{subj_str}.png'))
    return True

def main():
    csv_path = '/sci-it/projects/sarang-lab/desmond/MRIbreastseg/birads/sagittal_utah/uu_mip_tumor_and_birads.csv'
    out_dir = '/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/utah_sagittal_results'
    artifact_dir = '/home/desmond/.gemini/antigravity-ide/brain/4483c341-c4f8-40dd-849a-99984da733db/images'
    
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    def categorize_ratio(val):
        if pd.isna(val): return np.nan
        if val < 0.2410: return 'A'
        elif val < 0.3160: return 'B'
        elif val < 0.4133: return 'C'
        else: return 'D'

    df['Left_Sagittal_Class'] = df['Left_Sagittal_Ratio'].apply(categorize_ratio)
    df = df.dropna(subset=['Left_Sagittal_Class'])
    df['Subject'] = df['Subject'].apply(lambda x: str(x).zfill(6))
    
    # We must pick subjects based on their Sagittal classification
    examples = {}
    for cls in ['A', 'B', 'C', 'D']:
        cls_df = df[(df['Left_Sagittal_Class'] == cls) & (df['Subject'] != '622854')]
        if len(cls_df) > 0:
            examples[cls] = cls_df['Subject'].head(2).tolist()
            
    md_content = "# Utah Sagittal MIP Visuals\n\n"
    
    for cls, subjs in examples.items():
        md_content += f"## Category {cls}\n"
        for subj in subjs:
            print(f"Generating Sagittal visuals for {subj} (Class {cls})...")
            success = process_subject(subj, cls, out_dir, artifact_dir)
            if success:
                md_content += f"**Subject: {subj}**\n\n"
                md_content += f"![{subj} Sagittal Visual](/home/desmond/.gemini/antigravity-ide/brain/4483c341-c4f8-40dd-849a-99984da733db/images/utah_sagittal_example_class_{cls}_{subj}.png)\n\n"
            
    # Markdown update removed as it is handled by Walkthrough artifact
        
    print("Sagittal Visual generation complete.")

if __name__ == '__main__':
    main()
