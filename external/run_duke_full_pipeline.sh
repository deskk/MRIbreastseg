#!/bin/bash
set -e

echo "======================================"
echo "Starting DUKE Full Fusion Pipeline"
echo "======================================"

source ~/.bashrc

# Backup original config and symlink the DUKE config
echo "Symlinking DUKE config..."
if [ -f config.json ]; then
    mv config.json config_backup.json
fi
ln -s external/config_duke.json config.json

# Ensure cleanup on exit
cleanup() {
    echo "Restoring original config..."
    rm config.json
    if [ -f config_backup.json ]; then
        mv config_backup.json config.json
    fi
}
trap cleanup EXIT

# --- Phase 0 ---
echo "[1/4] Phase 0: Data Preparation (DICOM -> NIfTI)"
conda activate breastseg
# python external/prepare_duke_nifti.py

# --- Phase 1 ---
echo "[2/4] Phase 1: Torso Segmentation & Midline Split"
python phase1_breastdivider/scripts/run_phase1_inference.py
python phase1_breastdivider/scripts/split_breasts_midline.py

# --- Phase 2 ---
echo "[3/4] Phase 2: Tumor Segmentation & Noise Filtering"
conda activate mamamia
python phase2_mama-mia/scripts/run_phase2_mamamia.py
python phase2_mama-mia/scripts/postprocess_filter_noise.py

# --- Phase 3 ---
# echo "[4/4] Phase 3: FGT/DV Segmentation"
# conda activate fgt_env
# cd phase3_fgt-vessel/3D-Breast-FGT-and-Blood-Vessel-Segmentation
# python scripts/preprocess_batch.py
# python predict.py -c breast -i inference_data/batch/images -s inference_data/batch/preds_breast -p trained_models/breast_model.pth
# python predict.py -c dv -i inference_data/batch/images -m inference_data/batch/preds_breast -s inference_data/batch/preds_dv -p trained_models/dv_model.pth
# python scripts/export_nifti_batch.py
# cd ../../

# --- Phase 4 ---
# echo "[5/4] Phase 4: Derive Skin and Fat Masks"
# conda activate fgt_env
# python phase4_skin-fat/scripts/derive_skin_fat.py

# --- Phase 5 ---
# echo "[6/4] Phase 5: Final Multi-Label Fusion"
# python phase5_fusion/scripts/fuse_pipeline.py
# python phase5_fusion/scripts/calculate_birads.py

echo "======================================"
echo "DUKE Pipeline Execution Completed!"
echo "======================================"
