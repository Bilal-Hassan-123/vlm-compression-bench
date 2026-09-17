#!/bin/bash
#SBATCH --job-name=check_qwen_tokens
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=check_qwen_tokens_%j.log
#SBATCH --error=check_qwen_tokens_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/conda/envs/vlmeval

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

cd /scratch/bh2863/fastv_project/VLMEvalKit
python check_qwen_token_counts.py
