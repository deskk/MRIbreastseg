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

REGISTERED_DIR = config["PHASE0"]["REGISTERED_OUTPUT_DIR"]
MASK_DIR = config["PHASE1"]["OUTPUT_MASK_DIR"]
LEFT_OUT_DIR = config["PHASE1"]["OUTPUT_LEFT_DIR"]
RIGHT_OUT_DIR = config["PHASE1"]["OUTPUT_RIGHT_DIR"]

LEFT_LABEL = 1
RIGHT_LABEL = 2

def process_subject(subj):
    # SKIP LOGIC
    expected_left = os.path.join(LEFT_OUT_DIR, subj, f"{subj}_BreastDivider_Mask.nii.gz")
    expected_right = os.path.join(RIGHT_OUT_DIR, subj, f"{subj}_BreastDivider_Mask.nii.gz")
    if os.path.exists(expected_left) and os.path.exists(expected_right):
        logging.info(f"Skipping {subj}, midline splitting already completed.")
        return

    mask_path = os.path.join(MASK_DIR, subj, f"{subj}_BreastDivider_Mask.nii.gz")
    if not os.path.exists(mask_path):
        logging.warning(f"No Phase 1 mask found for {subj}. Skipping.")
        return

    # Load Mask
    mask_img = sitk.ReadImage(mask_path)
    mask_size = mask_img.GetSize()
    mask_spacing = mask_img.GetSpacing()

    # Find the midline between Label 1 (Left) and Label 2 (Right)
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_img)

    if not label_stats.HasLabel(LEFT_LABEL) or not label_stats.HasLabel(RIGHT_LABEL):
        logging.error(f"Missing left or right breast labels in mask for {subj}. Skipping.")
        return

    bbox_left = label_stats.GetBoundingBox(LEFT_LABEL)
    bbox_right = label_stats.GetBoundingBox(RIGHT_LABEL)

    # ITK Bounding Box format: (startX, startY, startZ, sizeX, sizeY, sizeZ)
    center_x_left = bbox_left[0] + bbox_left[3] / 2.0
    center_x_right = bbox_right[0] + bbox_right[3] / 2.0

    mid_x = int((center_x_left + center_x_right) / 2)
    size_x = mask_size[0]

    # Determine which way the X-axis is oriented
    if center_x_left < center_x_right:
        left_slice = slice(0, mid_x)
        right_slice = slice(mid_x, size_x)
    else:
        left_slice = slice(mid_x, size_x)
        right_slice = slice(0, mid_x)

    # Gather all images for this subject
    subj_reg_dir = os.path.join(REGISTERED_DIR, subj)
    if not os.path.exists(subj_reg_dir):
        logging.error(f"Registered directory not found for {subj}. Skipping.")
        return

    mri_files = glob.glob(os.path.join(subj_reg_dir, "*.nii.gz"))
    if not mri_files:
        logging.error(f"No MRI sequences found for {subj}. Skipping.")
        return

    all_files_to_split = mri_files + [mask_path]

    # Validation Pass: Ensure ALL images have identical dimensions before splitting
    for f in all_files_to_split:
        img_info = sitk.ReadImage(f)
        if img_info.GetSize() != mask_size:
            logging.error(f"Catastrophic Size Mismatch: {f} has size {img_info.GetSize()}, mask has {mask_size}. Skipping {subj}.")
            return
        # Using a small tolerance for spacing checks due to floating point precision
        for sp1, sp2 in zip(img_info.GetSpacing(), mask_spacing):
            if abs(sp1 - sp2) > 1e-4:
                logging.error(f"Catastrophic Spacing Mismatch: {f} spacing {img_info.GetSpacing()}, mask has {mask_spacing}. Skipping {subj}.")
                return

    # Splitting Pass
    left_out_subj_dir = os.path.join(LEFT_OUT_DIR, subj)
    right_out_subj_dir = os.path.join(RIGHT_OUT_DIR, subj)
    os.makedirs(left_out_subj_dir, exist_ok=True)
    os.makedirs(right_out_subj_dir, exist_ok=True)

    for f in all_files_to_split:
        img = sitk.ReadImage(f)
        filename = os.path.basename(f)

        # Note: SimpleITK slicing uses [X, Y, Z] order
        left_img = img[left_slice, :, :]
        right_img = img[right_slice, :, :]

        sitk.WriteImage(left_img, os.path.join(left_out_subj_dir, filename))
        sitk.WriteImage(right_img, os.path.join(right_out_subj_dir, filename))

    logging.info(f"Successfully split all sequences and mask for {subj}.")

def main():
    os.makedirs(LEFT_OUT_DIR, exist_ok=True)
    os.makedirs(RIGHT_OUT_DIR, exist_ok=True)

    if not os.path.exists(MASK_DIR):
        logging.error("Mask directory does not exist.")
        return

    subjects = [d for d in os.listdir(MASK_DIR) if os.path.isdir(os.path.join(MASK_DIR, d))]
    logging.info(f"Found {len(subjects)} subjects in phase1_mask to process.")

    for subj in subjects:
        process_subject(subj)

if __name__ == "__main__":
    main()
