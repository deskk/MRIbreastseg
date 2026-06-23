import os
import glob
import csv
import json
import numpy as np
import SimpleITK as sitk

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def get_padded_bbox(image, label, padding=20):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(image)
    if not stats.HasLabel(label):
        return None
    bbox = list(stats.GetBoundingBox(label))
    # bbox: (startX, startY, startZ, sizeX, sizeY, sizeZ)
    for i in range(3):
        start = max(0, bbox[i] - padding)
        end = min(image.GetSize()[i], bbox[i] + bbox[i+3] + padding)
        bbox[i] = start
        bbox[i+3] = end - start
    return tuple(bbox)

def crop_image(image, bbox):
    cropper = sitk.RegionOfInterestImageFilter()
    cropper.SetIndex(bbox[0:3])
    cropper.SetSize(bbox[3:6])
    return cropper.Execute(image)

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
    fusion_left_dir = config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"]
    fusion_right_dir = config["PHASE5"]["OUTPUT_FUSION_RIGHT_DIR"]
    tumor_presence_csv = config["PHASE5"]["OUTPUT_TUMOR_PRESENCE_CSV"]

    sides_info = {
        'left': {'label': 1, 'out_mri': out_mri_left, 'out_fusion': fusion_left_dir},
        'right': {'label': 2, 'out_mri': out_mri_right, 'out_fusion': fusion_right_dir}
    }

    os.makedirs(out_mri_left, exist_ok=True)
    os.makedirs(out_mri_right, exist_ok=True)
    os.makedirs(fusion_left_dir, exist_ok=True)
    os.makedirs(fusion_right_dir, exist_ok=True)
    
    if not os.path.exists(bd_dir):
        print("No Phase 1 output found.")
        return
        
    subjects = [d for d in os.listdir(bd_dir) if os.path.isdir(os.path.join(bd_dir, d))]
    
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
            mri_full_path = os.path.join(mri_dir, subj, f"{subj}_DYN_registered.nii.gz")
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

            for side, info in sides_info.items():
                print(f"  Side: {side}")
                label = info['label']
                
                # 1. Calculate Padded BBox
                bbox = get_padded_bbox(bd_full, label)
                if not bbox:
                    print(f"    No label {label} found in BD Mask.")
                    continue
                    
                # 2. Crop everything
                bd_cropped = crop_image(bd_full, bbox)
                mri_cropped = crop_image(mri_full, bbox)
                
                # Zero out contralateral label in BD mask
                bd_arr = sitk.GetArrayFromImage(bd_cropped)
                bd_arr[bd_arr != label] = 0
                
                # Arrays for fusion
                fat_arr = np.zeros_like(bd_arr)
                if fat_full:
                    fat_cropped = crop_image(fat_full, bbox)
                    fat_arr = sitk.GetArrayFromImage(fat_cropped)
                    fat_arr[bd_arr == 0] = 0
                    
                skin_arr = np.zeros_like(bd_arr)
                if skin_full:
                    skin_cropped = crop_image(skin_full, bbox)
                    skin_arr = sitk.GetArrayFromImage(skin_cropped)
                    skin_arr[bd_arr == 0] = 0
                    
                dv_arr = np.zeros_like(bd_arr)
                if dv_full:
                    # Resample dv to bd grid in case of slight origin mismatch
                    dv_resampled = sitk.Resample(dv_full, bd_cropped, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, bd_cropped.GetPixelID())
                    dv_arr = sitk.GetArrayFromImage(dv_resampled)
                    dv_arr[bd_arr == 0] = 0
                    
                tumor_arr = np.zeros_like(bd_arr)
                if tumor_full:
                    tumor_resampled = sitk.Resample(tumor_full, bd_cropped, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, bd_cropped.GetPixelID())
                    tumor_arr = sitk.GetArrayFromImage(tumor_resampled)
                    tumor_arr[bd_arr == 0] = 0
                    if np.any(tumor_arr == 1):
                        tumor_status[side] = True

                # 3. Fuse Hierarchically
                fused_arr = np.zeros_like(bd_arr, dtype=np.uint8)
                fused_arr[fat_arr == 1] = 1
                fused_arr[(bd_arr == label) & (dv_arr == 2)] = 3
                fused_arr[(bd_arr == label) & (dv_arr == 1)] = 4
                fused_arr[tumor_arr == 1] = 5
                fused_arr[skin_arr == 1] = 2

                # 4. Save Final Fused Mask and Single-Breast MRI
                fused_img = sitk.GetImageFromArray(fused_arr)
                fused_img.CopyInformation(bd_cropped)
                
                out_fusion_subj = os.path.join(info['out_fusion'], subj)
                os.makedirs(out_fusion_subj, exist_ok=True)
                sitk.WriteImage(fused_img, os.path.join(out_fusion_subj, f"{subj}_final_fusion.nii.gz"))
                
                out_mri_subj = os.path.join(info['out_mri'], subj)
                os.makedirs(out_mri_subj, exist_ok=True)
                sitk.WriteImage(mri_cropped, os.path.join(out_mri_subj, f"{subj}_DYN_cropped.nii.gz"))
            
            # Log CSV
            csv_writer.writerow([subj, tumor_status['left'], tumor_status['right']])
            csv_file.flush()
            
    print("Batch processing complete. Tumor presence saved to tumor_presence.csv.")

if __name__ == "__main__":
    main()
