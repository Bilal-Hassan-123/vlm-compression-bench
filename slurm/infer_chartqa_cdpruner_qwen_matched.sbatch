#!/bin/bash
#SBATCH --job-name=qwen_cdpruner_matched
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=qwen_cdpruner_matched_%j.out
#SBATCH --error=qwen_cdpruner_matched_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/envs/HiPrune-Qwen
module load gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

cd /scratch/bh2863/fastv_project/VLMEvalKit
python infer_chartqa_cdpruner_qwen_matched.py
