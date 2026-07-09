import os
import glob
import csv
import json
import numpy as np
import SimpleITK as sitk

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def resample_image(image, is_label=False, target_spacing=(0.25, 0.25, 0.25)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
        
    return resample.Execute(image)

def main():
    config = load_config()
    
    mri_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]
    bd_dir = config["PHASE1"]["OUTPUT_MASK_DIR"]
    tumor_dir = config["PHASE2"]["OUTPUT_FULL_DIR"]
    fgt_vessel_dir = config["PHASE3"]["OUTPUT_FGT_VESSEL_DIR"]
    fat_dir = config["PHASE4"]["OUTPUT_FAT_DIR"]
    skin_dir = config["PHASE4"]["OUTPUT_SKIN_DIR"]
    
    out_mri_left = config["PHASE5"]["OUTPUT_SPLIT_MRI_LEFT_DIR"]
    out_mri_right = config["PHASE5"]["OUTPUT_SPLIT_MRI_RIGHT_DIR"]
    
    # We output everything directly to the native-res folder
    fusion_left_native_dir = config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"] + "_native"
    fusion_right_native_dir = config["PHASE5"]["OUTPUT_FUSION_RIGHT_DIR"] + "_native"
    
    # Keep the old variables for compatibility, but point them to native dir
    fusion_left_dir = fusion_left_native_dir
    fusion_right_dir = fusion_right_native_dir
    
    tumor_presence_csv = config["PHASE5"]["OUTPUT_TUMOR_PRESENCE_CSV"]

    sides_info = {
        'left': {'label': 1, 'out_mri': out_mri_left, 'out_fusion': fusion_left_dir},
        'right': {'label': 2, 'out_mri': out_mri_right, 'out_fusion': fusion_right_dir}
    }

    os.makedirs(out_mri_left, exist_ok=True)
    os.makedirs(out_mri_right, exist_ok=True)
    os.makedirs(fusion_left_dir, exist_ok=True)
    os.makedirs(fusion_right_dir, exist_ok=True)
    os.makedirs(fusion_left_native_dir, exist_ok=True)
    os.makedirs(fusion_right_native_dir, exist_ok=True)

    
    if not os.path.exists(bd_dir):
        print("No Phase 1 output found.")
        return
        
    subjects = [d for d in os.listdir(bd_dir) if os.path.isdir(os.path.join(bd_dir, d))]
    
    test_subjects = config.get("TEST_SUBJECTS", [])
    if test_subjects:
        subjects = [s for s in subjects if s in test_subjects]
    
    with open(tumor_presence_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Subject', 'Left_Tumor_Present', 'Right_Tumor_Present'])

        for subj in subjects:
            print(f"Phase 5 [Split & Fusion] Processing Subject {subj}...")
            
            # Load full torso BD Mask
            bd_files = glob.glob(os.path.join(bd_dir, subj, "*_BreastDivider_Mask.nii.gz"))
            if not bd_files:
                continue
            bd_full = sitk.ReadImage(bd_files[0])
            
            # Load full torso MRI
            mri_full_path = os.path.join(mri_dir, subj, f"{subj}_DYN1_registered.nii.gz")
            if not os.path.exists(mri_full_path):
                # Fallback to PRE if DYN1 is missing
                mri_full_path = os.path.join(mri_dir, subj, f"{subj}_PRE_registered.nii.gz")
                if not os.path.exists(mri_full_path):
                    continue
            mri_full = sitk.ReadImage(mri_full_path)
            
            # Load full torso FGT-Vessel
            dv_full_path = os.path.join(fgt_vessel_dir, subj, f"{subj}_dv_mask.nii.gz")
            dv_full = sitk.ReadImage(dv_full_path) if os.path.exists(dv_full_path) else None
            
            # Load full torso Tumor
            tumor_files = glob.glob(os.path.join(tumor_dir, subj, "*_MAMAMIA_Mask.nii.gz"))
            tumor_full = sitk.ReadImage(tumor_files[0]) if tumor_files else None
            
            # Load full torso Fat and Skin
            fat_path = os.path.join(fat_dir, subj, f"{subj}_fat_mask.nii.gz")
            skin_path = os.path.join(skin_dir, subj, f"{subj}_skin_mask.nii.gz")
            fat_full = sitk.ReadImage(fat_path) if os.path.exists(fat_path) else None
            skin_full = sitk.ReadImage(skin_path) if os.path.exists(skin_path) else None

            tumor_status = {'left': False, 'right': False}

            # Find bounding boxes
            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(bd_full)

            if not label_stats.HasLabel(1) or not label_stats.HasLabel(2):
                print(f"Missing left or right breast labels in mask for {subj}. Skipping.")
                continue

            for side, info in sides_info.items():
                print(f"  Side: {side}")
                label = info['label']
                
                # 1. Fuse Hierarchically in full resolution first
                # Convert images to arrays for masking
                bd_arr_full = sitk.GetArrayFromImage(bd_full)
                bd_arr_full[bd_arr_full != label] = 0
                
                fat_arr = np.zeros_like(bd_arr_full)
                if fat_full:
                    fat_arr = sitk.GetArrayFromImage(fat_full)
                    fat_arr[bd_arr_full == 0] = 0
                    
                skin_arr = np.zeros_like(bd_arr_full)
                if skin_full:
                    skin_arr = sitk.GetArrayFromImage(skin_full)
                    skin_arr[bd_arr_full == 0] = 0
                    
                dv_arr = np.zeros_like(bd_arr_full)
                if dv_full:
                    dv_resampled = sitk.Resample(dv_full, bd_full, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, bd_full.GetPixelID())
                    dv_arr = sitk.GetArrayFromImage(dv_resampled)
                    dv_arr[bd_arr_full == 0] = 0
                    
                tumor_arr = np.zeros_like(bd_arr_full)
                if tumor_full:
                    tumor_resampled = sitk.Resample(tumor_full, bd_full, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, bd_full.GetPixelID())
                    tumor_arr = sitk.GetArrayFromImage(tumor_resampled)
                    tumor_arr[bd_arr_full == 0] = 0
                    if np.any(tumor_arr == 1):
                        tumor_status[side] = True

                # Fuse
                fused_arr_full = np.zeros_like(bd_arr_full, dtype=np.uint8)
                fused_arr_full[fat_arr == 1] = 1
                fused_arr_full[(bd_arr_full == label) & (dv_arr == 2)] = 3
                fused_arr_full[(bd_arr_full == label) & (dv_arr == 1)] = 4
                fused_arr_full[tumor_arr == 1] = 5
                fused_arr_full[skin_arr == 1] = 2

                fused_full_img = sitk.GetImageFromArray(fused_arr_full)
                fused_full_img.CopyInformation(bd_full)

                # 2. Crop using bounding box
                bbox = label_stats.GetBoundingBox(label)
                margin = 5
                size = bd_full.GetSize()
                x_start = max(0, bbox[0] - margin)
                y_start = max(0, bbox[1] - margin)
                z_start = max(0, bbox[2] - margin)
                x_end = min(size[0], bbox[0] + bbox[3] + margin)
                y_end = min(size[1], bbox[1] + bbox[4] + margin)
                z_end = min(size[2], bbox[2] + bbox[5] + margin)
                
                my_slice = (slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end))
                
                fused_cropped = fused_full_img[my_slice[0], my_slice[1], my_slice[2]]
                mri_cropped = mri_full[my_slice[0], my_slice[1], my_slice[2]]

                # 3. Save Final Fused Mask and Single-Breast MRI (Native Res)
                out_fusion_subj = os.path.join(info['out_fusion'], subj)
                os.makedirs(out_fusion_subj, exist_ok=True)
                sitk.WriteImage(fused_cropped, os.path.join(out_fusion_subj, f"{subj}_final_fusion.nii.gz"))
                
                out_mri_subj = os.path.join(info['out_mri'], subj)
                os.makedirs(out_mri_subj, exist_ok=True)
                sitk.WriteImage(mri_cropped, os.path.join(out_mri_subj, f"{subj}_DYN_cropped.nii.gz"))
            
            # Log CSV
            csv_writer.writerow([subj, tumor_status['left'], tumor_status['right']])
            csv_file.flush()
            
    print("Batch processing complete. Tumor presence saved to tumor_presence.csv.")

if __name__ == "__main__":
    main()
