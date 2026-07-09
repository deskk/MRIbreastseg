import os, json
import pandas as pd
dataset_dir = '/local/scratch/scratch-hd/desmond/datasets/DUKE-full/duke_breast_cancer_mri'
excel_path = '/local/scratch/scratch-hd/desmond/datasets/DUKE-full/Breast-Cancer-MRI-filepath_filename-mapping.xlsx'
df = pd.read_excel(excel_path)
mapping = {}
for _, row in df.iterrows():
    c = [p for p in str(row['classic_path']).split('/') if p]
    d = [p for p in str(row['descriptive_path']).split('/') if p]
    if len(c) >= 2 and len(d) >= 2:
        mapping[c[-2]] = d[-2]
all_subjects = sorted([d for d in os.listdir(dataset_dir) if d.startswith('Breast_MRI_')])
for subj in all_subjects[:10]:
    subj_dir = os.path.join(dataset_dir, subj)
    studies = [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
    if not studies: continue
    study_dir = os.path.join(subj_dir, studies[0])
    series_uids = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
    pre, post = [], []
    for suid in series_uids:
        desc = mapping.get(suid, '').lower()
        if not desc or 'sub' in desc or 'mask' in desc or 'segmentation' in desc or 't2' in desc or ('t1' in desc and 'dyn' not in desc and 'post' not in desc): continue
        if 'dyn' in desc or 'dynamic' in desc or 'vibrant' in desc:
            if 'pre' in desc: pre.append(desc)
            elif 'ph' not in desc and not any(c.isdigit() for c in desc.replace('3d', '')): pre.append(desc)
            else: post.append(desc)
    print(f'{subj}:\n  PRE: {pre}\n  POST: {post}')
