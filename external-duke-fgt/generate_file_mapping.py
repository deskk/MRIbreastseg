import os
import csv
import glob

def main():
    metadata_path = '/local/scratch/scratch-hd/desmond/datasets/DUKE-fgtvessels/subjects/manifest-1768961156411/metadata.csv'
    dataset_root = '/local/scratch/scratch-hd/desmond/datasets/DUKE-fgtvessels/subjects/manifest-1768961156411'
    output_dir = '/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external-duke-fgt/duke_outputs/phase0'
    csv_out_path = '/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/external-duke-fgt/duke_file_mapping.csv'

    subjects = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row['Subject ID']
            if subj not in subjects:
                subjects[subj] = []
            subjects[subj].append(row)

    mapping_data = []

    for subj, rows in subjects.items():
        pre_dir = None
        pre_desc = None
        dyn_dir = None
        dyn_desc = None
        
        # Find PRE
        for row in rows:
            desc = row['Series Description'].lower()
            if 'pre' in desc:
                pre_dir = row['File Location']
                pre_desc = row['Series Description']
                break
        if not pre_dir:
            for row in rows:
                desc = row['Series Description'].lower()
                if 't1' in desc and 'dyn' not in desc and 'post' not in desc:
                    pre_dir = row['File Location']
                    pre_desc = row['Series Description']
                    break
                    
        # Find DYN
        dyn_candidates = []
        for row in rows:
            desc = row['Series Description'].lower()
            if ('dyn' in desc or 'dynamic' in desc or 'vibrant' in desc) and 'pre' not in desc and 't2' not in desc:
                dyn_candidates.append(row)
                
        # Sort by description
        dyn_candidates.sort(key=lambda x: x['Series Description'].lower())
        if dyn_candidates:
            dyn_dir = dyn_candidates[0]['File Location']
            dyn_desc = dyn_candidates[0]['Series Description']

        # Check NIfTI files
        subj_out = os.path.join(output_dir, subj)
        out_pre = os.path.join(subj_out, f"{subj}_PRE_registered.nii.gz")
        out_dyn = os.path.join(subj_out, f"{subj}_DYN_registered.nii.gz")
        out_t1 = os.path.join(subj_out, f"{subj}_AX_3D_T1_registered.nii.gz")

        has_pre_nifti = os.path.exists(out_pre)
        has_dyn_nifti = os.path.exists(out_dyn)
        has_t1_nifti = os.path.exists(out_t1)

        mapping_data.append({
            'Subject_ID': subj,
            'Original_PRE_Desc': pre_desc,
            'Original_DYN_Desc': dyn_desc,
            'PRE_NIfTI_Exists': has_pre_nifti,
            'DYN_NIfTI_Exists': has_dyn_nifti,
            'T1_NIfTI_Exists': has_t1_nifti,
            'Original_PRE_DICOM_Path': os.path.normpath(os.path.join(dataset_root, pre_dir.lstrip('./'))) if pre_dir else "None",
            'Original_DYN_DICOM_Path': os.path.normpath(os.path.join(dataset_root, dyn_dir.lstrip('./'))) if dyn_dir else "None",
            'Generated_PRE_Path': out_pre,
            'Generated_DYN_Path': out_dyn,
            'Generated_T1_Path': out_t1
        })

    # Sort by subject ID
    mapping_data.sort(key=lambda x: x['Subject_ID'])

    with open(csv_out_path, 'w', newline='') as f:
        fieldnames = ['Subject_ID', 'Original_PRE_Desc', 'Original_DYN_Desc', 'PRE_NIfTI_Exists', 'DYN_NIfTI_Exists', 'T1_NIfTI_Exists', 'Original_PRE_DICOM_Path', 'Original_DYN_DICOM_Path', 'Generated_PRE_Path', 'Generated_DYN_Path', 'Generated_T1_Path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in mapping_data:
            writer.writerow(row)

    print(f"Generated file mapping for {len(mapping_data)} subjects. Saved to {csv_out_path}")

if __name__ == '__main__':
    main()
