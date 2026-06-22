import os
import csv
import glob
import SimpleITK as sitk

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
            pre_img = reader.Execute()
            sitk.WriteImage(pre_img, out_pre)

            # Save an identical copy for T1 structural placeholder if needed by Phase 1
            sitk.WriteImage(pre_img, out_t1)

            # Read DYN
            full_dyn_dir = os.path.normpath(os.path.join(dataset_root, dyn_dir.lstrip('./')))
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(full_dyn_dir)
            if not dicom_names:
                print(f"  No DICOM files found in DYN dir: {full_dyn_dir}")
                continue
            reader.SetFileNames(dicom_names)
            dyn_img = reader.Execute()
            sitk.WriteImage(dyn_img, out_dyn)
            
            success_count += 1
            print(f"  Successfully converted {subj}.")
        except Exception as e:
            print(f"  Error processing {subj}: {e}")

    print(f"Finished processing {success_count} / {len(subjects)} subjects.")

if __name__ == '__main__':
    main()
