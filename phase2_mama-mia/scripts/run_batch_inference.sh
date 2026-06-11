#!/bin/bash

# Source conda environment
source /local/scratch/scratch-hd/desmond/miniconda3/etc/profile.d/conda.sh
conda activate mamamia

# Set environment variables
export nnUNet_raw='/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/nnUNet/nnunetv2/nnUNet_raw'
export nnUNet_preprocessed=''
export nnUNet_results='/local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/nnUNet/nnunetv2/nnUNet_results'

echo "====================================="
echo "1. PREPROCESSING ALL SUBJECTS"
echo "====================================="
python /local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/src/pipeline/preprocess_test_baseline.py

echo "====================================="
echo "2. RUNNING nnUNet INFERENCE"
echo "====================================="
nnUNetv2_predict -i /local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/nnUNet/nnunetv2/nnUNet_raw/Dataset102_Test/imagesTs -o /local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/nnUNet/nnunetv2/nnUNet_raw/Dataset102_Test/output_masks -d 101 -c 3d_fullres -device cuda --save_probabilities

echo "====================================="
echo "3. POSTPROCESSING ALL SUBJECTS"
echo "====================================="
python /local/scratch/scratch-hd/desmond/research/Summer2026/MRIbreastseg/phase2_mama-mia/src/pipeline/postprocess_test_baseline.py

echo "====================================="
echo "BATCH PIPELINE COMPLETE"
echo "====================================="
