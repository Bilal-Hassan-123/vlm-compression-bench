#!/bin/bash
#SBATCH --job-name=chartqa_infer_fastv
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=infer_chartqa_fastv_%j.out
#SBATCH --error=infer_chartqa_fastv_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate fastv
cd /scratch/bh2863/fastv_project/VLMEvalKit

python infer_chartqa_fastv.py
