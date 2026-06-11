#!/bin/bash
set -e

echo "======================================"
echo "Starting Full Production Pipeline"
echo "======================================"

source ~/.bashrc

# --- Phase 0 ---
echo "[1/12] Phase 0: Raw Data Preprocessing & Registration"
# Note: batch_registration uses argparse, or you can update it to use config.
# For now we assume the user will configure the input/output directly if needed, or we just run it as configured.
conda activate breastseg
python phase0_preprocessing/scripts/batch_registration.py --input_dir /local/scratch/scratch-hd/desmond/projects/BreastCancerRad/raw_data/13-09-22/pre_processing/NFB\ Breast\ MRI --output_dir data/registered --workers 20

# --- Phase 1 ---
echo "[2/12] Phase 1: Torso Segmentation"
python phase1_breastdivider/scripts/run_phase1_inference.py

echo "[3/12] Phase 1: Midline Split"
python phase1_breastdivider/scripts/split_breasts_midline.py

# --- Phase 2 ---
echo "[4/12] Phase 2: Tumor Segmentation"
conda activate mamamia
python phase2_mama-mia/scripts/run_phase2_mamamia.py

echo "[5/12] Phase 2: Noise Filtering"
python phase2_mama-mia/scripts/postprocess_filter_noise.py

# --- Phase 3 ---
echo "[6/12] Phase 3: FGT/DV Preprocessing"
conda activate fgt_env
cd phase3_fgt-vessel/3D-Breast-FGT-and-Blood-Vessel-Segmentation
python scripts/preprocess_batch.py

echo "[7/12] Phase 3: Predict Breast Mask"
# In a real environment, read models from config, here we just use relative path inside the directory
python predict.py -c breast -i inference_data/batch/images -s inference_data/batch/preds_breast -p trained_models/breast_model.pth

echo "[8/12] Phase 3: Predict Dense-Vessel Mask"
python predict.py -c dv -i inference_data/batch/images -m inference_data/batch/preds_breast -s inference_data/batch/preds_dv -p trained_models/dv_model.pth

echo "[9/12] Phase 3: Export to NIfTI"
python scripts/export_nifti_batch.py
cd ../../

# --- Phase 4 ---
echo "[10/12] Phase 4: Derive Skin and Fat Masks"
conda activate fgt_env
python phase4_skin-fat/scripts/derive_skin_fat.py

# --- Phase 5 ---
echo "[11/12] Phase 5: Final Multi-Label Fusion"
python phase5_fusion/scripts/fuse_pipeline.py

echo "[12/12] Phase 5: BI-RADS Calculation & Figure Generation"
python phase5_fusion/scripts/calculate_birads.py
python phase5_fusion/scripts/generate_figures.py

echo "======================================"
echo "Pipeline Execution Completed!"
echo "======================================"
