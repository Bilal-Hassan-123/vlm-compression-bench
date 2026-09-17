#!/bin/bash
#SBATCH --job-name=vz_h0
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=vz_h0_%j.out
#SBATCH --error=vz_h0_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/envs/HiPrune-Qwen
module load gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

cd /scratch/bh2863/fastv_project/VLMEvalKit
export VZ_DOM=0.207
export VZ_CTX=0.016
export VZ_TAG=h0
python infer_vz_split.py
