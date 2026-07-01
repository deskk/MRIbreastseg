'''
Run as: python /local/scratch/scratch-hd/desmond/research/breastdivider/scripts/batch_registration.py --input_dir /local/scratch/scratch-hd/desmond/projects/BreastCancerRad/raw_data/13-09-22/pre_processing/NFB Breast MRI --output_dir /local/scratch/scratch-hd/desmond/projects/BreastCancerRad/raw_data/13-09-22/pre_processing/NFB Breast MRI_registered --workers 20 --test

this is for breast mri registration to make dynamic images registered to t1, and t2 registered to t1, and pre registered to t1, 
t1 registered to t1. all outputted images will be in the same orientation as t1 and the same resolution as t1.

the dynamic images are multi-frame, so we will need to register each frame to t1. 

for the structural images, we will use rigid + affine registration.

for the dynamic images, we will use rigid + affine + deformable registration.

the pre registered image is also a dynamic image, so we will need to register it to t1.

all outputted images will be in the same orientation as t1 and the same resolution as t1.

the dynamic images are multi-frame, so we will need to register each frame to t1.

for the structural images, we will use rigid + affine registration.

for the dynamic images, we will use rigid + affine + deformable registration.

the pre registered image is also a dynamic image, so we will need to register it to t1.

all outputted images will be in the same orientation as t1 and the same resolution as t1.
'''


import os
# Prevent ITK and OpenMP from spawning multiple threads per process
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import glob
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import ants
import numpy as np

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(processName)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def process_subject(subject_dir, output_root):
    subject_id = os.path.basename(os.path.normpath(subject_dir))
    out_dir = os.path.join(output_root, subject_id)
    os.makedirs(out_dir, exist_ok=True)
    
    logger = logging.getLogger()
    
    # Check if already processed
    expected_files_count = len(glob.glob(os.path.join(out_dir, "*.nii.gz")))
    if expected_files_count >= 5: # 5 files are expected to be outputted
        logger.info(f"{subject_id}: Already processed, skipping.")
        return True
    
    # 1. Find the required sequences
    # Fixed Anchor: *PRE*.nii.gz
    pre_files = glob.glob(os.path.join(subject_dir, "*PRE*.nii.gz"))
    if not pre_files:
        logger.error(f"{subject_id}: Missing PRE (Fixed Anchor) image.")
        return False
    fixed_path = pre_files[0]
    
    # Moving Dyn: The *first* dynamic phase after PRE.
    # Exclude PRE and SUB from the search.
    all_dyn_files = glob.glob(os.path.join(subject_dir, "*DYN*.nii.gz"))
    # The variations could be DYNAMIC or DYANAMIC so we match *DYN*
    dyn_candidates = [f for f in all_dyn_files if "PRE" not in f.upper() and "SUB" not in f.upper()]
    dyn_candidates.sort() # Alphabetical sort ensuring the first one is selected (e.g. s000005 vs s000007)
    if not dyn_candidates:
        logger.error(f"{subject_id}: Missing post-contrast DYN image.")
        return False
    moving_dyn_path = dyn_candidates[0]
    
    # T1 Structural
    t1_files = glob.glob(os.path.join(subject_dir, "*AX*3D*T1*.nii.gz"))
    t1_files = [f for f in t1_files if "DYN" not in f.upper()]
    if not t1_files:
        logger.error(f"{subject_id}: Missing AX 3D T1 image.")
        return False
    t1_path = t1_files[0]
    
    # T2 Structural
    t2_files = glob.glob(os.path.join(subject_dir, "*T2*SPACE*STIR*.nii.gz"))
    if not t2_files:
        logger.error(f"{subject_id}: Missing T2 SPACE STIR image.")
        return False
    t2_path = t2_files[0]

    try:
        logger.info(f"{subject_id}: Loading images...")
        fixed_img = ants.image_read(fixed_path)
        moving_dyn_img = ants.image_read(moving_dyn_path)
        t1_img = ants.image_read(t1_path)
        t2_img = ants.image_read(t2_path)
        
        # --- Step 1: Intra-Sequence (DCE) Registration ---
        logger.info(f"{subject_id}: Registering DYN to PRE (SyN)...")
        # Deformable SyN with MI (Mattes mutual information is the default metric for SyN in ANTsPy)
        reg_dyn = ants.registration(
            fixed=fixed_img, 
            moving=moving_dyn_img, 
            type_of_transform='SyN',
            aff_metric='mattes',
            syn_metric='mattes'
        )
        reg_dyn_img = reg_dyn['warpedmovout']
        
        # Generate new subtraction map
        logger.info(f"{subject_id}: Computing subtraction map...")
        sub_img_array = reg_dyn_img.numpy() - fixed_img.numpy()
        sub_img = fixed_img.new_image_like(sub_img_array)
        
        # --- Step 2: Inter-Sequence (Structural) Registration ---
        logger.info(f"{subject_id}: Registering T1 to PRE (SyNRA)...")
        reg_t1 = ants.registration(
            fixed=fixed_img,
            moving=t1_img,
            type_of_transform='SyNRA',
            aff_metric='mattes',
            syn_metric='mattes'
        )
        reg_t1_img = reg_t1['warpedmovout']

        logger.info(f"{subject_id}: Registering T2 to PRE (SyNRA)...")
        reg_t2 = ants.registration(
            fixed=fixed_img,
            moving=t2_img,
            type_of_transform='SyNRA',
            aff_metric='mattes',
            syn_metric='mattes'
        )
        reg_t2_img = reg_t2['warpedmovout']
        
        # --- Step 3: Saving Outputs ---
        logger.info(f"{subject_id}: Saving outputs...")
        
        # Helper to format output filepath
        def get_out_path(orig_path, suffix="_registered"):
            base_name = os.path.basename(orig_path)
            # handle .nii.gz safely
            if base_name.endswith('.nii.gz'):
                name_part = base_name[:-7]
                ext_part = '.nii.gz'
            else:
                name_part, ext_part = os.path.splitext(base_name)
            return os.path.join(out_dir, f"{name_part}{suffix}{ext_part}")

        # Save the fixed image reference directly to output dir as well for convenience
        ants.image_write(fixed_img, os.path.join(out_dir, os.path.basename(fixed_path)))

        out_dyn_path = get_out_path(moving_dyn_path, "_registered")
        ants.image_write(reg_dyn_img, out_dyn_path)
        
        # Subtraction is based on the dynamic registered image naming but with SUB suffix
        out_sub_path = get_out_path(moving_dyn_path, "_registered_SUB")
        ants.image_write(sub_img, out_sub_path)
        
        out_t1_path = get_out_path(t1_path, "_registered")
        ants.image_write(reg_t1_img, out_t1_path)
        
        out_t2_path = get_out_path(t2_path, "_registered")
        ants.image_write(reg_t2_img, out_t2_path)
        
        logger.info(f"{subject_id}: Processing complete.")
        return True

    except Exception as e:
        logger.error(f"{subject_id}: Failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch Registration of Breast MRI sequences using ANTsPy")
    parser.add_argument("--input_dir", required=True, help="Directory containing original subject folders")
    parser.add_argument("--output_dir", required=True, help="Directory to save registered subject folders")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 4), help="Number of parallel workers")
    parser.add_argument("--test", action="store_true", help="Run only on process on one subject for testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "registration_batch.log"))
    logger = logging.getLogger()
    logger.info(f"Starting batch registration")
    
    subject_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    subject_dirs.sort()
    
    if args.test:
        logger.info("TEST MODE: Running only on the first subject.")
        subject_dirs = subject_dirs[:1]

    logger.info(f"Found {len(subject_dirs)} subjects to process. Using {args.workers} workers.")

    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_subject = {executor.submit(process_subject, sdir, args.output_dir): sdir for sdir in subject_dirs}
        
        for future in as_completed(future_to_subject):
            sdir = future_to_subject[future]
            sid = os.path.basename(sdir)
            try:
                success = future.result()
                if success:
                    success_count += 1
            except Exception as exc:
                logger.error(f"{sid} generated an exception: {exc}")

    logger.info(f"Batch processing finished. Successfully processed {success_count}/{len(subject_dirs)} subjects.")

if __name__ == "__main__":
    main()
