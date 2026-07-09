#!/bin/bash
#SBATCH --job-name=breast_mri_pipeline
#SBATCH --account=common
#SBATCH --partition=general-gpu
#SBATCH --array=1-50                 # Spawns 50 parallel tasks (adjust based on dataset size)
#SBATCH --gpus=1                     # 1 GPU allocated to EACH of the 50 tasks
#SBATCH --cpus-per-task=10           # 10 CPU cores allocated to EACH task
#SBATCH --mem=64G                    # 64GB of RAM allocated to EACH task
#SBATCH --time=12:00:00              # Max walltime per task (Hours:Minutes:Seconds)
#SBATCH --output=logs/job_%A_task_%a.txt

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
# Conda initialization is handled directly inside run_slurm_pipeline.sh

# CRITICAL FIX FOR PHASE 0 (ANTs): 
# This forces ANTs/ITK to utilize all 10 requested CPU cores instead of just 1.
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_CPUS_PER_TASK

# ==========================================
# 2. DYNAMIC PATHING FOR THE JOB ARRAY
# ==========================================
# $SLURM_ARRAY_TASK_ID dynamically changes for each of the 50 jobs (1, 2, 3... up to 50)
INPUT_CHUNK="slurm/data/metadata_chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt"
OUTPUT_DIR="slurm/results/chunk_${SLURM_ARRAY_TASK_ID}"

echo "Starting Slurm Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing input directory: $INPUT_CHUNK"
echo "Using CPU threads for ANTs: $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"

# ==========================================
# 3. RUN THE PIPELINE
# ==========================================
# 'srun' safely executes your script within the allocated cluster resources
srun bash slurm/run_slurm_pipeline.sh --input_chunk $INPUT_CHUNK --output_dir $OUTPUT_DIR