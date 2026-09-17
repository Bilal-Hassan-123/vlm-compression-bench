#!/bin/bash
#SBATCH --job-name=chartqa_export
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=export_chartqa_%j.out
#SBATCH --error=export_chartqa_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate vlmeval
cd /scratch/bh2863/fastv_project/VLMEvalKit

python export_chartqa.py
