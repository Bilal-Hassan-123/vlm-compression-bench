#!/bin/bash
#SBATCH --job-name=llavanext_baseline
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=llavanext_baseline_%j.out
#SBATCH --error=llavanext_baseline_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/conda/envs/vlmeval

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

cd /scratch/bh2863/fastv_project/VLMEvalKit
python infer_chartqa_llavanext_baseline.py
