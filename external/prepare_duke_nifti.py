import os
import csv
import glob
import SimpleITK as sitk
import ants

def main():
    metadata_path = '/local/scratch/scratch-hd/desmond/datasets/DUKE-fgtvessels/subjects/manifest-1768961156411/metadata.csv'
    dataset_root = '/local/scratch/scratch-hd/desmond/datasets/DUKE-fgtvessels/subjects/manifest-1768961156411'
    output_dir = '/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external/duke_outputs/phase0'

    os.makedirs(output_dir, exist_ok=True)

    subjects = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row['Subject ID']
            if subj not in subjects:
                subjects[subj] = []
            subjects[subj].append(row)

    print(f"Found {len(subjects)} subjects in metadata.")

    success_count = 0
    # To test one subject as requested, we can break after the first success, or process all. The script will be run by run_duke_full_pipeline.sh which might just run on whatever is there.
    # The user said "the goal is to test one subject from end to end". Let's restrict it to the first subject only for testing.
    for subj, rows in subjects.items():
        print(f"Processing {subj}...")
        
        pre_dir = None
        dyn_dir = None
        
        # Find PRE
        for row in rows:
            desc = row['Series Description'].lower()
            if 'pre' in desc:
                pre_dir = row['File Location']
                break
        if not pre_dir:
            for row in rows:
                desc = row['Series Description'].lower()
                if 't1' in desc and 'dyn' not in desc and 'post' not in desc:
                    pre_dir = row['File Location']
                    break
                    
        # Find DYN
        dyn_candidates = []
        for row in rows:
            desc = row['Series Description'].lower()
            if ('dyn' in desc or 'dynamic' in desc or 'vibrant' in desc) and 'pre' not in desc and 't2' not in desc:
                dyn_candidates.append(row)
                
        # Sort by description to prioritize 1st pass or smallest number
        dyn_candidates.sort(key=lambda x: x['Series Description'].lower())
        if dyn_candidates:
            dyn_dir = dyn_candidates[0]['File Location']
            
        if not pre_dir or not dyn_dir:
            print(f"  Missing required sequences for {subj}. PRE: {bool(pre_dir)}, DYN: {bool(dyn_dir)}")
            continue

        subj_out = os.path.join(output_dir, subj)
        os.makedirs(subj_out, exist_ok=True)
        
        out_pre = os.path.join(subj_out, f"{subj}_PRE_registered.nii.gz")
        out_dyn = os.path.join(subj_out, f"{subj}_DYN_registered.nii.gz")
        out_t1 = os.path.join(subj_out, f"{subj}_AX_3D_T1_registered.nii.gz")
        
        if os.path.exists(out_pre) and os.path.exists(out_dyn):
            print(f"  Already processed {subj}.")
            success_count += 1
            break # Just one subject!

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

            # Read DYN
            full_dyn_dir = os.path.normpath(os.path.join(dataset_root, dyn_dir.lstrip('./')))
            dicom_names = reader.GetGDCMSeriesFileNames(full_dyn_dir)
            if not dicom_names:
                print(f"  No DICOM files found in DYN dir: {full_dyn_dir}")
                continue
            reader.SetFileNames(dicom_names)
            dyn_img_sitk = reader.Execute()
            
            # Save PRE directly
            sitk.WriteImage(pre_img_sitk, out_pre)
            sitk.WriteImage(pre_img_sitk, out_t1) # Placeholder

            print(f"  Performing ANTs SyN Registration for {subj}...")
            # Use ants.image_read to load directly from the written file to avoid direction matrix shape issues
            fixed_img = ants.image_read(out_pre)
            
            # Write DYN to a temp file and read it back
            temp_dyn = os.path.join(subj_out, f"{subj}_DYN_temp.nii.gz")
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
            reg_dyn_img = reg_dyn['warpedmovout']
            
            # Save warped directly using ANTs
            ants.image_write(reg_dyn_img, out_dyn)
            
            # Clean up temp
            if os.path.exists(temp_dyn):
                os.remove(temp_dyn)

            success_count += 1
            print(f"  Successfully registered and converted {subj}.")
            break # Just one subject for end-to-end test
        except Exception as e:
            print(f"  Error processing {subj}: {e}")

    print(f"Finished processing {success_count} / {len(subjects)} subjects.")

if __name__ == '__main__':
    main()
