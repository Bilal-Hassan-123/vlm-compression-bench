#!/bin/bash
#SBATCH --job-name=eval_qwen_baseline
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=eval_qwen_baseline_%j.out
#SBATCH --error=eval_qwen_baseline_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate vlmeval

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData

cd /scratch/bh2863/fastv_project/VLMEvalKit
python run.py --data ChartQA_TEST --model hiprune_baseline_qwen2_5_vl_3b --mode eval
