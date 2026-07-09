import os
import tempfile
# Force ANTs and Python to use the spacious scratch drive for temp files
scratch_tmp = '/local/scratch/scratch-hd/desmond/tmp'
os.makedirs(scratch_tmp, exist_ok=True)
os.environ['TMPDIR'] = scratch_tmp
tempfile.tempdir = scratch_tmp

import json
import pydicom
import SimpleITK as sitk
import ants

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)

def get_sorted_dynamic_series(study_dir, series_uids):
    series_info = []
    
    for suid in series_uids:
        series_path = os.path.join(study_dir, suid)
        dcms = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        if not dcms:
            continue
        
        dcm_path = os.path.join(series_path, dcms[0])
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            acq_time = getattr(ds, 'AcquisitionTime', None)
            if acq_time is None:
                acq_time = getattr(ds, 'SeriesTime', None)
                
            series_desc = getattr(ds, 'SeriesDescription', 'UNKNOWN').lower()
            
            series_info.append({
                'uid': suid,
                'desc': series_desc,
                'time': acq_time or '000000',
                'path': series_path
            })
        except Exception as e:
            print(f"  Failed to read DICOM {dcm_path}: {e}")
            
    # Filter for dynamic series only
    dyn_series = []
    for s in series_info:
        desc = s['desc']
        if 'sub' in desc or 'mask' in desc or 'segmentation' in desc or 't2' in desc or ('t1' in desc and 'dyn' not in desc and 'post' not in desc):
            continue
            
        if 'dyn' in desc or 'dynamic' in desc or 'vibrant' in desc:
            dyn_series.append(s)
            
    # Sort chronologically by time
    dyn_series.sort(key=lambda x: str(x['time']))
    
    # If no dynamic series, try to find a t1 pre
    if not dyn_series:
        for s in series_info:
            desc = s['desc']
            if 't1' in desc and 'dyn' not in desc and 't2' not in desc and 'sub' not in desc and 'mask' not in desc:
                dyn_series.append(s)
                break
                
    return dyn_series

def main():
    config = load_config()
    raw_input_dir = config["PHASE0"]["RAW_INPUT_DIR"]
    dataset_root = os.path.join(raw_input_dir, 'duke_breast_cancer_mri')
    output_dir = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]

    os.makedirs(output_dir, exist_ok=True)
    
    test_subjects = config.get("TEST_SUBJECTS", [])
    if not test_subjects:
        all_subjects = sorted([d for d in os.listdir(dataset_root) if d.startswith('Breast_MRI_')])
        test_subjects = all_subjects[:5] # Target 5 subjects
        
    print(f"Targeting {len(test_subjects)} subjects.")

    success_count = 0
    for subj in test_subjects:
        print(f"\nProcessing {subj}...")
        
        subj_dir = os.path.join(dataset_root, subj)
        if not os.path.exists(subj_dir):
            print(f"  Directory not found for {subj}: {subj_dir}")
            continue
            
        studies = [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
        if not studies:
            print(f"  No studies found for {subj}")
            continue
            
        study_uid = studies[0] 
        study_dir = os.path.join(subj_dir, study_uid)
        
        series_uids = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
        
        dyn_series = get_sorted_dynamic_series(study_dir, series_uids)
        
        if len(dyn_series) < 2:
            print(f"  Missing required sequences for {subj}. Total dynamic phases found: {len(dyn_series)}")
            continue
            
        pre_cands = [dyn_series[0]]
        post_cands = dyn_series[1:]
        
        print(f"  [Report] Found series for {subj}:")
        print(f"    - PRE Series: {pre_cands[0]['time']} - {pre_cands[0]['desc']} ({pre_cands[0]['uid']})")
        for i, post in enumerate(post_cands):
            print(f"    - POST Series {i+1}: {post['time']} - {post['desc']} ({post['uid']})")

        pre_dir = pre_cands[0]['path']
        post_dirs = [cand['path'] for cand in post_cands]

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
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(pre_dir)
            if not dicom_names:
                print(f"  No DICOM files found in PRE dir: {pre_dir}")
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
            resampler.SetInterpolator(sitk.sitkBSpline)
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
                dicom_names = reader.GetGDCMSeriesFileNames(dyn_dir)
                if not dicom_names:
                    print(f"  No DICOM files found in DYN dir: {dyn_dir}")
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

    print(f"Finished processing {success_count} / {len(test_subjects)} subjects.")

if __name__ == '__main__':
    main()
