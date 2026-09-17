#!/bin/bash
#SBATCH --job-name=eval_hiprune_qwen
#SBATCH --partition=preempt
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=eval_hiprune_qwen_%j.out

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate vlmeval

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

python run.py --data ChartQA_TEST --model hiprune_qwen2_5_vl_3b --mode eval
