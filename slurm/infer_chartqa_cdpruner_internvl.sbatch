#!/bin/bash
#SBATCH --job-name=internvl_cdpruner
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=internvl_cdpruner_%j.out
#SBATCH --error=internvl_cdpruner_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/envs/HiPrune-Qwen

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

cd /scratch/bh2863/fastv_project/VLMEvalKit
python infer_chartqa_cdpruner_internvl.py
