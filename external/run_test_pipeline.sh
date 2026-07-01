#!/bin/bash
set -e

echo "======================================"
echo "Starting DUKE Test Pipeline"
echo "======================================"

source ~/.bashrc

# Backup original config and symlink the test config
echo "Symlinking Test config..."
if [ -f config.json ]; then
    mv config.json config_backup.json
fi
ln -s external/config_test.json config.json

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
if [ -f external/.test_phase0.done ]; then
    echo "[1/6] Phase 0: Data Preparation & Registration already completed. Skipping..."
else
    echo "[1/6] Phase 0: Data Preparation & Registration (DICOM -> NIfTI)"
    conda activate breastseg
    python external/prepare_duke_nifti.py
    touch external/.test_phase0.done
fi

# --- Phase 1 ---
if [ -f external/.test_phase1.done ]; then
    echo "[2/6] Phase 1: Torso Segmentation already completed. Skipping..."
else
    echo "[2/6] Phase 1: Torso Segmentation (Full Bilateral)"
    conda activate breastseg
    python phase1_breastdivider/scripts/run_phase1_inference.py
    touch external/.test_phase1.done
fi

# --- Phase 2 ---
if [ -f external/.test_phase2.done ]; then
    echo "[3/6] Phase 2: Tumor Segmentation & Noise Filtering already completed. Skipping..."
else
    echo "[3/6] Phase 2: Tumor Segmentation & Noise Filtering (Full Bilateral)"
    conda activate mamamia
    python phase2_mama-mia/scripts/run_phase2_mamamia.py
    python phase2_mama-mia/scripts/postprocess_filter_noise.py
    touch external/.test_phase2.done
fi

# --- Phase 3 ---
if [ -f external/.test_phase3.done ]; then
    echo "[4/6] Phase 3: FGT/DV Segmentation already completed. Skipping..."
else
    echo "[4/6] Phase 3: FGT/DV Segmentation (Full Bilateral)"
    conda activate fgt_env
    cd phase3_fgt-vessel/3D-Breast-FGT-and-Blood-Vessel-Segmentation
    python scripts/preprocess_batch.py
    python predict.py -c breast -i inference_data/batch/images -s inference_data/batch/preds_breast -p trained_models/breast_model.pth
    python predict.py -c dv -i inference_data/batch/images -m inference_data/batch/preds_breast -s inference_data/batch/preds_dv -p trained_models/dv_model.pth
    python scripts/export_nifti_batch.py
    cd ../../
    touch external/.test_phase3.done
fi

# --- Phase 4 ---
if [ -f external/.test_phase4.done ]; then
    echo "[5/6] Phase 4: Derive Skin and Fat Masks already completed. Skipping..."
else
    echo "[5/6] Phase 4: Derive Skin and Fat Masks (Full Bilateral)"
    conda activate fgt_env
    python phase4_skin-fat/scripts/derive_skin_fat.py
    touch external/.test_phase4.done
fi

# --- Phase 5 ---
if [ -f external/.test_phase5.done ]; then
    echo "[6/6] Phase 5: Split to Unilateral & Final Multi-Label Fusion already completed. Skipping..."
else
    echo "[6/6] Phase 5: Split to Unilateral & Final Multi-Label Fusion"
    conda activate breastseg
    python phase5_fusion/scripts/fuse_pipeline.py
    python phase5_fusion/scripts/calculate_birads.py
    touch external/.test_phase5.done
fi

echo "======================================"
echo "DUKE Test Pipeline Execution Completed!"
echo "======================================"
