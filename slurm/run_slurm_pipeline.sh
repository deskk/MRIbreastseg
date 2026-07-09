#!/bin/bash
set -e

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_chunk) INPUT_CHUNK="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$INPUT_CHUNK" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --input_chunk <chunk.csv> --output_dir <path>"
    exit 1
fi

echo "======================================"
echo "Starting DUKE Full Fusion Pipeline (SLURM ISOLATED)"
echo "Input Chunk: $INPUT_CHUNK"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================"

mkdir -p "$OUTPUT_DIR"
CONFIG_PATH="$OUTPUT_DIR/config_duke.json"

# Create a customized config_duke.json for this chunk
# 1. Replace the local workspace path with the SLURM shared workspace path
# 2. Replace the global external-duke-full/duke_outputs with our job-specific OUTPUT_DIR
sed "s|/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg|/sci-it/projects/sarang-lab/desmond/MRIbreastseg|g" external-duke-full/config_duke_full.json \
    | sed "s|/sci-it/projects/sarang-lab/desmond/MRIbreastseg/external-duke-full/duke_outputs|$OUTPUT_DIR|g" \
    | sed "s|/local/scratch/scratch-hd/desmond/datasets/DUKE-full|/sci-it/projects/sarang-lab/desmond/datasets/DUKE-full|g" > "$CONFIG_PATH"

# Dynamically populate TEST_SUBJECTS from the input chunk file
python3 -c "import json; c=json.load(open('$CONFIG_PATH')); c['TEST_SUBJECTS']=[x.strip() for x in open('$INPUT_CHUNK') if x.strip()]; json.dump(c, open('$CONFIG_PATH','w'), indent=4)"

export SLURM_DUKE_CONFIG=$(readlink -f "$CONFIG_PATH")
export SLURM_DUKE_METADATA=$(readlink -f "$INPUT_CHUNK")
export SLURM_DUKE_PHASE0_OUT=$(readlink -f "$OUTPUT_DIR/phase0")
export SLURM_DUKE_PHASE3_TEMP=$(readlink -f "$OUTPUT_DIR/phase3_temp")

# Initialize Conda properly for the slurm job
source /sci-it/projects/sarang-lab/desmond/miniconda3/etc/profile.d/conda.sh

# Prevent nnUNet warnings by exporting dummy paths (we use predict_from_modelfolder which doesn't need these)
export nnUNet_raw="${OUTPUT_DIR}/nnunet_tmp/raw"
export nnUNet_preprocessed="${OUTPUT_DIR}/nnunet_tmp/preprocessed"
export nnUNet_results="${OUTPUT_DIR}/nnunet_tmp/results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# --- Phase 0 ---
if [ -f "$OUTPUT_DIR/.phase0.done" ]; then
    echo "[1/6] Phase 0: Data Preparation & Registration already completed. Skipping..."
else
    echo "[1/6] Phase 0: Data Preparation & Registration (DICOM -> NIfTI)"
    conda activate breastseg
    python external-duke-full/prepare_duke_full_nifti.py
    touch "$OUTPUT_DIR/.phase0.done"
fi

# --- Phase 1 ---
if [ -f "$OUTPUT_DIR/.phase1.done" ]; then
    echo "[2/6] Phase 1: Torso Segmentation already completed. Skipping..."
else
    echo "[2/6] Phase 1: Torso Segmentation (Full Bilateral)"
    conda activate breastseg
    python pipeline/phase1_breastdivider/scripts/run_phase1_inference.py
    touch "$OUTPUT_DIR/.phase1.done"
fi

# --- Phase 2 ---
if [ -f "$OUTPUT_DIR/.phase2.done" ]; then
    echo "[3/6] Phase 2: Tumor Segmentation & Noise Filtering already completed. Skipping..."
else
    echo "[3/6] Phase 2: Tumor Segmentation (Full Bilateral)"
    conda activate mamamia
    python pipeline/phase2_mama-mia/scripts/run_phase2_mamamia.py
    touch "$OUTPUT_DIR/.phase2.done"
fi

# --- Phase 3 ---
if [ -f "$OUTPUT_DIR/.phase3.done" ]; then
    echo "[4/6] Phase 3: FGT/DV Segmentation already completed. Skipping..."
else
    echo "[4/6] Phase 3: FGT/DV Segmentation (Full Bilateral)"
    conda activate fgt_env
    # Create temp directory for Phase 3 intermediary data
    mkdir -p "$SLURM_DUKE_PHASE3_TEMP/images" "$SLURM_DUKE_PHASE3_TEMP/preds_breast" "$SLURM_DUKE_PHASE3_TEMP/preds_dv"
    
    cd pipeline/phase3_fgt-vessel/3D-Breast-FGT-and-Blood-Vessel-Segmentation
    PYTHONPATH=. python scripts/preprocess_batch.py
    
    python predict.py -c breast -i "$SLURM_DUKE_PHASE3_TEMP/images" -s "$SLURM_DUKE_PHASE3_TEMP/preds_breast" -p trained_models/breast_model.pth
    python predict.py -c dv -i "$SLURM_DUKE_PHASE3_TEMP/images" -m "$SLURM_DUKE_PHASE3_TEMP/preds_breast" -s "$SLURM_DUKE_PHASE3_TEMP/preds_dv" -p trained_models/dv_model.pth
    
    python scripts/export_nifti_batch.py
    cd ../../../
    touch "$OUTPUT_DIR/.phase3.done"
fi

# --- Phase 4 ---
if [ -f "$OUTPUT_DIR/.phase4.done" ]; then
    echo "[5/6] Phase 4: Derive Skin and Fat Masks already completed. Skipping..."
else
    echo "[5/6] Phase 4: Derive Skin and Fat Masks (Full Bilateral)"
    conda activate fgt_env
    python pipeline/phase4_skin-fat/scripts/derive_skin_fat.py
    touch "$OUTPUT_DIR/.phase4.done"
fi

# --- Phase 5 ---
if [ -f "$OUTPUT_DIR/.phase5.done" ]; then
    echo "[6/6] Phase 5: Split to Unilateral & Final Multi-Label Fusion already completed. Skipping..."
else
    echo "[6/6] Phase 5: Split to Unilateral & Final Multi-Label Fusion"
    conda activate breastseg
    python pipeline/phase5_fusion/scripts/fuse_pipeline.py
    python pipeline/phase5_fusion/scripts/calculate_birads.py
    touch "$OUTPUT_DIR/.phase5.done"
fi

echo "======================================"
echo "DUKE Pipeline SLURM Chunk Completed!"
echo "======================================"

echo "Cleaning up temporary isolated environments..."
rm -rf "${OUTPUT_DIR}/nnunet_tmp"
rm -rf "${OUTPUT_DIR}/phase3_temp"
echo "Cleanup complete."
