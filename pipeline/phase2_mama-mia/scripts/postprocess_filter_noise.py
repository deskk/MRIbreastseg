import os
import glob
import logging
import SimpleITK as sitk


import json
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)
config = load_config()


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OUTPUT_DIR = config["PHASE2"]["OUTPUT_FULL_DIR"]

# Minimum volume threshold in mm^3. 
# Since our images are resampled to 1x1x1 mm, 1 voxel = 1 mm^3.
# 50 mm^3 is a very safe threshold for clinical noise (lesions smaller than ~4-5mm).
MIN_VOLUME_MM3 = 50.0

def filter_noise(mask_path):
    mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    
    # Find all disconnected components in the mask
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_img = cc_filter.Execute(mask_img)
    
    # Calculate statistics (like volume) for each component
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled_img)
    
    labels_to_keep = []
    
    for label in stats.GetLabels():
        # Get physical volume in mm^3
        volume = stats.GetPhysicalSize(label)
        if volume >= MIN_VOLUME_MM3:
            labels_to_keep.append(label)
        else:
            logging.info(f"Removing noise component (Volume: {volume:.1f} mm^3) in {os.path.basename(mask_path)}")
            
    # If no labels pass the threshold, return a perfectly blank mask with correct metadata
    if not labels_to_keep:
        empty_mask = sitk.Image(mask_img.GetSize(), sitk.sitkUInt8)
        empty_mask.CopyInformation(mask_img)
        return empty_mask
        
    # Create a new mask preserving only the valid components
    cleaned_mask = sitk.Image(mask_img.GetSize(), sitk.sitkUInt8)
    cleaned_mask.CopyInformation(mask_img)
    
    for label in labels_to_keep:
        single_label_mask = sitk.BinaryThreshold(
            labeled_img, 
            lowerThreshold=label, 
            upperThreshold=label, 
            insideValue=1, 
            outsideValue=0
        )
        cleaned_mask = sitk.Or(cleaned_mask, single_label_mask)
        
    return cleaned_mask

def main():
    logging.info(f"Starting Connected Component Volumetric Filtering (Threshold: {MIN_VOLUME_MM3} mm^3)")
    if os.path.exists(OUTPUT_DIR):
        mask_files = glob.glob(os.path.join(OUTPUT_DIR, "*", "*_MAMAMIA_Mask.nii.gz"))
        
        for mask_path in mask_files:
            cleaned_mask = filter_noise(mask_path)
            # Overwrite the original mask with the cleaned version
            sitk.WriteImage(cleaned_mask, mask_path)
            
    logging.info("Noise filtering complete.")

if __name__ == "__main__":
    main()
