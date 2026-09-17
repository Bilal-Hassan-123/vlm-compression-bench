#!/bin/bash
#SBATCH --job-name=chartqa_hiprune
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=chartqa_hiprune_%j.out
#SBATCH --error=chartqa_hiprune_%j.err
export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData
export PIP_CACHE_DIR=/scratch/bh2863/pip_cache
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/envs/HiPrune-LLaVA
cd /scratch/bh2863/fastv_project/VLMEvalKit
python infer_chartqa_hiprune.py
