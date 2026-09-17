#!/bin/bash
#SBATCH --job-name=infer_chartqapro_hiprune_qwen
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=infer_chartqapro_hiprune_qwen_%j.out

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/envs/HiPrune-Qwen

module load gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

python infer_chartqapro_hiprune_qwen.py
