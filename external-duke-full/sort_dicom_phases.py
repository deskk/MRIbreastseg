import os
import pydicom

def sort_subject_series(dataset_dir, subj_id):
    subj_dir = os.path.join(dataset_dir, subj_id)
    studies = [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
    if not studies:
        print(f"No studies found for {subj_id}")
        return
    
    study_dir = os.path.join(subj_dir, studies[0])
    series_dirs = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
    
    series_info = []
    
    for suid in series_dirs:
        series_path = os.path.join(study_dir, suid)
        dcms = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        if not dcms:
            continue
        
        # Read the first DICOM to get series info
        dcm_path = os.path.join(series_path, dcms[0])
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            acq_time = getattr(ds, 'AcquisitionTime', None)
            if acq_time is None:
                acq_time = getattr(ds, 'SeriesTime', None)
                
            series_desc = getattr(ds, 'SeriesDescription', 'UNKNOWN')
            
            series_info.append({
                'uid': suid,
                'desc': series_desc,
                'time': acq_time or '000000'
            })
        except Exception as e:
            print(f"Failed to read {dcm_path}: {e}")
            
    # Sort by time
    series_info.sort(key=lambda x: x['time'])
    
    print(f"Sorted Series for {subj_id}:")
    for i, s in enumerate(series_info):
        print(f"  {i+1}. {s['time']} - {s['desc']} (UID: {s['uid']})")

if __name__ == '__main__':
    dataset = '/local/scratch/scratch-hd/desmond/datasets/DUKE-full/duke_breast_cancer_mri'
    sort_subject_series(dataset, 'Breast_MRI_002')
    print('-' * 40)
    sort_subject_series(dataset, 'Breast_MRI_001')
