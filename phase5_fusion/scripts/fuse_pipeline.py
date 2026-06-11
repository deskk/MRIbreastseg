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

def main():
    config = load_config()
    
    phase1_left_dir = config["PHASE1"]["OUTPUT_LEFT_DIR"]
    phase1_right_dir = config["PHASE1"]["OUTPUT_RIGHT_DIR"]
    phase2_left_dir = config["PHASE2"]["OUTPUT_LEFT_DIR"]
    phase2_right_dir = config["PHASE2"]["OUTPUT_RIGHT_DIR"]
    fgt_vessel_dir = config["PHASE3"]["OUTPUT_FGT_VESSEL_DIR"]
    fat_dir = config["PHASE4"]["OUTPUT_FAT_DIR"]
    skin_dir = config["PHASE4"]["OUTPUT_SKIN_DIR"]
    
    fusion_left_dir = config["PHASE5"]["OUTPUT_FUSION_LEFT_DIR"]
    fusion_right_dir = config["PHASE5"]["OUTPUT_FUSION_RIGHT_DIR"]
    tumor_presence_csv = config["PHASE5"]["OUTPUT_TUMOR_PRESENCE_CSV"]

    sides_info = {
        'left': {
            'p1': phase1_left_dir,
            'p2': phase2_left_dir,
            'out': fusion_left_dir
        },
        'right': {
            'p1': phase1_right_dir,
            'p2': phase2_right_dir,
            'out': fusion_right_dir
        }
    }

    os.makedirs(fusion_left_dir, exist_ok=True)
    os.makedirs(fusion_right_dir, exist_ok=True)
    
    # Get all subjects from the FGT vessel dir (since it's full torso)
    if not os.path.exists(fgt_vessel_dir):
        print("No Phase 3 output found.")
        return
        
    subjects = [d for d in os.listdir(fgt_vessel_dir) if os.path.isdir(os.path.join(fgt_vessel_dir, d))]
    
    with open(tumor_presence_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Subject', 'Left_Tumor_Present', 'Right_Tumor_Present'])

        for subj in subjects:
            print(f"Phase 5 [Fusion] Processing Subject {subj}...")
            
            # Load Full Torso DV
            dv_full_path = os.path.join(fgt_vessel_dir, subj, f"{subj}_dv_mask.nii.gz")
            if not os.path.exists(dv_full_path):
                print(f"  Missing full-torso FGT-Vessel for {subj}")
                continue
            dv_full = sitk.ReadImage(dv_full_path)

            tumor_status = {'left': False, 'right': False}

            for side, dirs in sides_info.items():
                print(f"  Side: {side}")
                
                # Check if already fused
                if os.path.exists(os.path.join(dirs['out'], subj, f"{subj}_final_fusion.nii.gz")):
                    print(f"    Already processed.")
                    continue
                
                # Load Phase 1 BreastDivider
                p1_files = glob.glob(os.path.join(dirs['p1'], subj, "*_BreastDivider_Mask.nii.gz"))
                if not p1_files:
                    print(f"    Missing BreastDivider for {side}")
                    continue
                bd_img = sitk.ReadImage(p1_files[0])
                bd_arr = sitk.GetArrayFromImage(bd_img)
                target_label = 1 if side == 'left' else 2

                # Split FGT-Vessel to unilateral
                dv_resampled = sitk.Resample(dv_full, bd_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, bd_img.GetPixelID())
                dv_arr = sitk.GetArrayFromImage(dv_resampled)
                dv_arr[bd_arr == 0] = 0

                # Load Phase 4 Fat & Skin
                fat_path = os.path.join(fat_dir, side, subj, f"{subj}_fat_mask.nii.gz")
                skin_path = os.path.join(skin_dir, side, subj, f"{subj}_skin_mask.nii.gz")
                
                if not os.path.exists(fat_path) or not os.path.exists(skin_path):
                    print(f"    Missing Phase 4 Skin/Fat for {side}")
                    continue
                    
                fat_arr = sitk.GetArrayFromImage(sitk.ReadImage(fat_path))
                skin_arr = sitk.GetArrayFromImage(sitk.ReadImage(skin_path))

                # Load Phase 2 Tumor
                tumor_path = os.path.join(dirs['p2'], subj, f"{subj}_MAMAMIA_Mask.nii.gz")
                tumor_arr = np.zeros_like(bd_arr)
                if os.path.exists(tumor_path):
                    tumor_img = sitk.ReadImage(tumor_path)
                    tumor_resampled = sitk.Resample(tumor_img, bd_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, tumor_img.GetPixelID())
                    tumor_arr = sitk.GetArrayFromImage(tumor_resampled)
                    
                    if np.any(tumor_arr == 1):
                        tumor_status[side] = True

                # Fuse Hierarchically
                fused_arr = np.zeros_like(bd_arr, dtype=np.uint8)
                
                # Priority 1: Fat
                fused_arr[fat_arr == 1] = 1
                
                # Priority 2: FGT
                fused_arr[(bd_arr == target_label) & (dv_arr == 2)] = 3
                
                # Priority 3: Vessels
                fused_arr[(bd_arr == target_label) & (dv_arr == 1)] = 4
                
                # Priority 4: Tumor 
                fused_arr[tumor_arr == 1] = 5
                
                # Priority 5: Skin 
                fused_arr[skin_arr == 1] = 2

                # Save Final Fused Mask
                fused_img = sitk.GetImageFromArray(fused_arr)
                fused_img.CopyInformation(bd_img)
                
                out_dir = os.path.join(dirs['out'], subj)
                os.makedirs(out_dir, exist_ok=True)
                sitk.WriteImage(fused_img, os.path.join(out_dir, f"{subj}_final_fusion.nii.gz"))
            
            # Log CSV
            csv_writer.writerow([subj, tumor_status['left'], tumor_status['right']])
            csv_file.flush()
            
    print("Batch processing complete. Tumor presence saved to tumor_presence.csv.")

if __name__ == "__main__":
    main()
