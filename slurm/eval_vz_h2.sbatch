#!/bin/bash
#SBATCH --job-name=eval_vz_h2
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=eval_vz_h2_%j.log
#SBATCH --error=eval_vz_h2_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/conda/envs/vlmeval
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

cd /scratch/bh2863/fastv_project/VLMEvalKit
python run.py --data ChartQA_TEST --model visionzip_qwen2_5_vl_3b_h2 --mode eval
