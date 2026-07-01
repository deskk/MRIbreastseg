import os
import tempfile
# Force ANTs and Python to use the spacious scratch drive for temp files
scratch_tmp = '/local/scratch/scratch-hd/desmond/tmp'
os.makedirs(scratch_tmp, exist_ok=True)
os.environ['TMPDIR'] = scratch_tmp
tempfile.tempdir = scratch_tmp

import csv
import glob
import json
import SimpleITK as sitk
import ants

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config = load_config()
    raw_input_dir = config["PHASE0"]["RAW_INPUT_DIR"]
    dataset_root = os.path.join(raw_input_dir, 'subjects/manifest-1768961156411')
    metadata_path = os.path.join(dataset_root, 'metadata.csv')
    output_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]

    os.makedirs(output_dir, exist_ok=True)

    subjects = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row['Subject ID']
            if subj not in subjects:
                subjects[subj] = []
            subjects[subj].append(row)

    test_subjects = config.get("TEST_SUBJECTS", [])
    if test_subjects:
        subjects = {s: rows for s, rows in subjects.items() if s in test_subjects}

    print(f"Found {len(subjects)} subjects in metadata.")

    success_count = 0
    for subj, rows in subjects.items():
        print(f"Processing {subj}...")

        pre_cands = []
        post_cands = []
        
        for row in rows:
            desc = row['Series Description'].lower()
            if 'sub' in desc or 'mask' in desc or 'segmentation' in desc or 't2' in desc or ('t1' in desc and 'dyn' not in desc and 'post' not in desc):
                continue
            
            if 'pre' in desc:
                pre_cands.append(row)
            elif 'dyn' in desc or 'dynamic' in desc or 'vibrant' in desc:
                if 'ph' not in desc and not any(char.isdigit() for char in desc.replace('3d', '')):
                    pre_cands.append(row)
                else:
                    post_cands.append(row)
                    
        # Fallback for t1 if no pre_cands found
        if not pre_cands:
            t1_cands = [r for r in rows if 't1' in r['Series Description'].lower() and 'dyn' not in r['Series Description'].lower() and 't2' not in r['Series Description'].lower()]
            if t1_cands:
                pre_cands.append(t1_cands[0])

        if not pre_cands or not post_cands:
            print(f"  Missing required sequences for {subj}. PRE count: {len(pre_cands)}, POST count: {len(post_cands)}")
            continue
            
        pre_dir = pre_cands[0]['File Location']
        
        # Sort post_cands to ensure temporal order
        post_cands.sort(key=lambda x: x['Series Description'].lower())
        post_dirs = [cand['File Location'] for cand in post_cands]

        subj_out = os.path.join(output_dir, subj)
        os.makedirs(subj_out, exist_ok=True)

        out_pre = os.path.join(subj_out, f"{subj}_PRE_registered.nii.gz")
        out_t1 = os.path.join(subj_out, f"{subj}_AX_3D_T1_registered.nii.gz")

        # Check if all expected files are present
        all_dce_present = os.path.exists(out_pre)
        for i in range(len(post_dirs)):
            if not os.path.exists(os.path.join(subj_out, f"{subj}_DYN{i+1}_registered.nii.gz")):
                all_dce_present = False
                break
                
        if all_dce_present:
            print(f"  Already processed {subj}.")
            success_count += 1
            continue

        try:
            # Read PRE
            full_pre_dir = os.path.normpath(os.path.join(dataset_root, pre_dir.lstrip('./')))
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(full_pre_dir)
            if not dicom_names:
                print(f"  No DICOM files found in PRE dir: {full_pre_dir}")
                continue
            reader.SetFileNames(dicom_names)
            pre_img_sitk = reader.Execute()
            
            # Reorient to LPS
            pre_img_sitk = sitk.DICOMOrient(pre_img_sitk, 'LPS')
            
            # Resample to original UU dataset parameters [0.8854, 0.8854, 1.2]
            original_spacing = pre_img_sitk.GetSpacing()
            original_size = pre_img_sitk.GetSize()
            new_spacing = [0.8854, 0.8854, 1.2]
            
            new_size = [
                int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
            ]
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputOrigin(pre_img_sitk.GetOrigin())
            resampler.SetOutputDirection(pre_img_sitk.GetDirection())
            resampler.SetInterpolator(sitk.sitkBSpline) # High quality interpolation for MRI
            resampler.SetDefaultPixelValue(0)
            
            pre_img_sitk = resampler.Execute(pre_img_sitk)

            # Save PRE directly
            sitk.WriteImage(pre_img_sitk, out_pre)
            sitk.WriteImage(pre_img_sitk, out_t1) # Placeholder

            print(f"  Performing ANTs SyN Registration for {subj} ({len(post_dirs)} POST phases)...")
            fixed_img = ants.image_read(out_pre)

            for i, dyn_dir in enumerate(post_dirs):
                dyn_num = i + 1
                out_dyn = os.path.join(subj_out, f"{subj}_DYN{dyn_num}_registered.nii.gz")
                
                # Read DYN
                full_dyn_dir = os.path.normpath(os.path.join(dataset_root, dyn_dir.lstrip('./')))
                dicom_names = reader.GetGDCMSeriesFileNames(full_dyn_dir)
                if not dicom_names:
                    print(f"  No DICOM files found in DYN dir: {full_dyn_dir}")
                    continue
                reader.SetFileNames(dicom_names)
                dyn_img_sitk = reader.Execute()
                
                temp_dyn = os.path.join(subj_out, f"{subj}_DYN{dyn_num}_temp.nii.gz")
                sitk.WriteImage(dyn_img_sitk, temp_dyn)
                moving_dyn_img = ants.image_read(temp_dyn)
                
                # Deformable SyN with MI
                reg_dyn = ants.registration(
                    fixed=fixed_img,
                    moving=moving_dyn_img,
                    type_of_transform='SyN',
                    aff_metric='mattes',
                    syn_metric='mattes'
                )
                
                ants.image_write(reg_dyn['warpedmovout'], out_dyn)
                if os.path.exists(temp_dyn):
                    os.remove(temp_dyn)

            success_count += 1
            print(f"  Successfully registered and converted {subj}.")
        except Exception as e:
            print(f"  Error processing {subj}: {e}")

    print(f"Finished processing {success_count} / {len(subjects)} subjects.")

if __name__ == '__main__':
    main()
