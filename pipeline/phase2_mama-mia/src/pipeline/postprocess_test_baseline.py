import os
import glob
import SimpleITK as sitk

DATASET_PATH = "/local/scratch/scratch-hd/desmond/dataset/clean_data_registered"
NNUNET_OUTPUT = "/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/nnUNet/nnunetv2/nnUNet_raw/Dataset102_Test/output_masks"

def get_reference_image(subject_dir):
    # Just grab any T1/DYNAMIC sequence to use its geometry header
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    for f in all_files:
        if "t2" not in os.path.basename(f).lower() and "sub" not in os.path.basename(f).lower():
            return f
    return all_files[0]

def resample_to_reference(moving_image_sitk, reference_image_sitk):
    # Resample moving image (mask) to the exact geometry of reference_image_sitk
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image_sitk)
    # Using NearestNeighbor because this is a classification mask (0, 1)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # Also ensure background stays 0
    resampler.SetDefaultPixelValue(0)
    
    try:
        resampled_img = resampler.Execute(moving_image_sitk)
    except RuntimeError:
        # Fallback if direction mismatch somehow causes an error
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        resampler.SetOutputDirection(direction)
        moving_image_sitk.SetDirection(direction)
        resampled_img = resampler.Execute(moving_image_sitk)
        
    return resampled_img

def postprocess_subject(subject_id):
    subject_dir = os.path.join(DATASET_PATH, subject_id)
    pred_mask_path = os.path.join(NNUNET_OUTPUT, f"MAMAMIA_{subject_id}.nii.gz")
    
    if not os.path.exists(pred_mask_path):
        print(f"Skipping {subject_id}: prediction mask not found at {pred_mask_path}")
        return
    
    ref_image_path = get_reference_image(subject_dir)
    print(f"\nReverting mask for {subject_id} using reference: {os.path.basename(ref_image_path)}")
    
    ref_sitk = sitk.ReadImage(ref_image_path)
    pred_sitk = sitk.ReadImage(pred_mask_path, sitk.sitkUInt8) # masks are uint8
    
    resampled_mask_sitk = resample_to_reference(pred_sitk, ref_sitk)
    
    out_path = os.path.join(subject_dir, f"MAMAMIA_PREDICTION_MASK.nii.gz")
    sitk.WriteImage(resampled_mask_sitk, out_path)
    print(f"Saved aligned mask to: {out_path}")
    print(f"Original Shape: {ref_sitk.GetSize()} | Spacing: {ref_sitk.GetSpacing()}")
    print(f"Reverted Mask Shape: {resampled_mask_sitk.GetSize()} | Spacing: {resampled_mask_sitk.GetSpacing()}")

if __name__ == '__main__':
    all_dirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    for subj in sorted(all_dirs):
        postprocess_subject(subj)
