import os
import glob
import json
import logging
import subprocess
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BREASTDIVIDER_SCRIPT_PATH = "/local/scratch/scratch-hd/desmond/research/breastdivider/inference.py"
PADDING_MARGIN_MM = 20.0
TARGET_SPACING = (1.0, 1.0, 1.0)
BACKGROUND_LABEL = 0
LEFT_BREAST_LABEL = 1
RIGHT_BREAST_LABEL = 2

def get_dynamic_sequences(subject_dir: str) -> Tuple[str, List[str]]:
    """Identifies the PRE and sorted POST registration series logically natively."""
    all_files = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    pre_file = None
    post_files = []
    
    for f in all_files:
        filename = os.path.basename(f).lower()
        if "sub" in filename or "mask" in filename or "t2" in filename or ("t1" in filename and "dynamic" not in filename and "dyanamic" not in filename):
            continue
            
        if "pre" in filename:
            pre_file = f
        elif "dynamic" in filename or "dyanamic" in filename:
            post_files.append(f)
            
    # Sort post files iteratively
    post_files.sort(key=lambda x: int(os.path.basename(x).split(" ")[0][1:]))
    return pre_file, post_files

def compute_global_statistics(all_dce_files: List[str]) -> Tuple[float, float]:
    """
    Computes global Z-score statistics (Mean and Standard Deviation)
    from all sequences (PRE + POSTs) natively excluding the structural background array zeros.
    """
    all_pixels = []
    for f in all_dce_files:
        img_sitk = sitk.ReadImage(f, sitk.sitkFloat32)
        arr = sitk.GetArrayViewFromImage(img_sitk).ravel()
        arr = arr[arr != 0] # exclude geometric baseline
        all_pixels.append(arr)
        
    all_pixels = np.concatenate(all_pixels)
    global_mean = float(np.mean(all_pixels))
    global_std = float(np.std(all_pixels))
    
    logging.info(f"Global Normalization DCE Statistics -> Mean: {global_mean:.4f}, Std: {global_std:.4f}")
    return global_mean, global_std

def calculate_padded_bbox_physical(mask_image: sitk.Image, target_label: int, margin_mm: float) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_image)
    
    if not label_stats.HasLabel(target_label):
        raise ValueError(f"Label {target_label} is not present in the mask!")

    bbox = label_stats.GetBoundingBox(target_label)
    min_idx = list(bbox[:3])
    size = list(bbox[3:])
    max_idx = [min_idx[i] + size[i] - 1 for i in range(3)]
    
    img_size = mask_image.GetSize()
    spacing = mask_image.GetSpacing()
    
    padded_min_idx = []
    padded_size = []
    
    for i in range(3):
        # Expand the Y axis: Anteriorly (-Y) by 10mm, Posteriorly (+Y) by margin_mm (20mm)
        if i == 1:
            pad_min = int(np.ceil(10.0 / spacing[i]))
            pad_max = int(np.ceil(margin_mm / spacing[i]))
        else:
            # Zero padding for X (Left-Right) and Z (Inferior-Superior) to prevent bounding box overlap
            pad_min = 0
            pad_max = 0
            
        new_min = max(0, min_idx[i] - pad_min)
        new_max = min(img_size[i] - 1, max_idx[i] + pad_max)
        padded_min_idx.append(new_min)
        padded_size.append(new_max - new_min + 1)
        
    return tuple(padded_min_idx), tuple(padded_size)

def crop_normalize_and_resample(image: sitk.Image, start_index: Tuple[int, ...], extract_size: Tuple[int, ...], 
                                global_mean: float, global_std: float) -> Tuple[sitk.Image, Dict]:
    # 1. Truncate sequentially
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(start_index)
    roi_filter.SetSize(extract_size)
    cropped_img = roi_filter.Execute(image)
    
    # 2. Compile Physical Coordinate JSON metadata explicitly 
    metadata = {
        "Origin": cropped_img.GetOrigin(),
        "Spacing": cropped_img.GetSpacing(),
        "Direction": cropped_img.GetDirection(),
        "Size": cropped_img.GetSize()
    }
    
    # 3. Z-score mathematical normalization
    cropped_float = sitk.Cast(cropped_img, sitk.sitkFloat32)
    normalized_img = sitk.ShiftScale(cropped_float, shift=-global_mean, scale=1.0 / global_std)

    # 4. Isotropic Affine Mappings iteratively
    original_spacing = normalized_img.GetSpacing()
    original_size = normalized_img.GetSize()
    
    resampled_size = [
        int(np.round(sz * (spc / tgt_spc)))
        for sz, spc, tgt_spc in zip(original_size, original_spacing, TARGET_SPACING)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(tuple(resampled_size))
    resampler.SetOutputOrigin(normalized_img.GetOrigin())
    resampler.SetOutputDirection(normalized_img.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(normalized_img))))
    
    final_resampled_img = resampler.Execute(normalized_img)
    return final_resampled_img, metadata

def process_subject_sequence(subject_dir: str, output_base_dir: str, existing_mask_path: str = None):
    # Retrieve the phase sequences properly!
    pre_file, post_files = get_dynamic_sequences(subject_dir)
    if not post_files:
         raise Exception(f"No DCE POST phases cleanly found in {subject_dir}!")
         
    first_post_file = post_files[0]
    all_dce_files = post_files.copy()
    if pre_file:
         all_dce_files.append(pre_file)
         
    # Generate Phase Statistics safely
    mean, std = compute_global_statistics(all_dce_files)
    
    os.makedirs(output_base_dir, exist_ok=True)
    mask_path_tmp = os.path.join(output_base_dir, "breastdivider_LR_mask.nii.gz")
    
    if existing_mask_path and os.path.exists(existing_mask_path):
        import shutil
        shutil.copy(existing_mask_path, mask_path_tmp)
    elif not os.path.exists(mask_path_tmp):
        pass # Missing BreastDivider inference
        
    # Read the 1st POST sequence explicitly for the structural generation inferences!
    image_1st_post = sitk.ReadImage(first_post_file)
    mask = sitk.ReadImage(mask_path_tmp)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    regions = {"Left": LEFT_BREAST_LABEL, "Right": RIGHT_BREAST_LABEL}
    
    for side, label in regions.items():
        try:
            start_idx, ext_size = calculate_padded_bbox_physical(mask, label, margin_mm=PADDING_MARGIN_MM)
            # CRITICAL FIX: Pass the FIRST POST image for cropping natively!
            crop_img, metadata = crop_normalize_and_resample(
                image=image_1st_post, start_index=start_idx, extract_size=ext_size, 
                global_mean=mean, global_std=std
            )
            
            nifti_out_path = os.path.join(output_base_dir, f"{side.lower()}_crop_1x1x1.nii.gz")
            json_out_path = os.path.join(output_base_dir, f"{side.lower()}_crop_metadata.json")
            
            sitk.WriteImage(crop_img, nifti_out_path)
            with open(json_out_path, "w") as jf:
                json.dump(metadata, jf, indent=4)
                
        except Exception as e:
            logging.error(f"Failed extracting the {side} topological region: {e}")

if __name__ == "__main__":
    pass
