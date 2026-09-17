#!/bin/bash
#SBATCH --job-name=chartqa_check
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=chartqa_check_%j.out
#SBATCH --error=chartqa_check_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate vlmeval
cd /scratch/bh2863/fastv_project/VLMEvalKit

python -c "
from vlmeval.dataset import build_dataset
d = build_dataset('ChartQA_TEST')
print(d.data.columns.tolist())
print(d.data.head())
"
