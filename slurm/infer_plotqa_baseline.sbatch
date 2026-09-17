#!/bin/bash
#SBATCH --job-name=plotqa_infer_base
#SBATCH --partition=preempt
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=infer_plotqa_baseline_%j.out
#SBATCH --error=infer_plotqa_baseline_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate fastv
cd /scratch/bh2863/fastv_project/VLMEvalKit

python infer_plotqa_baseline.py
