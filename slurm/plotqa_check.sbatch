#!/bin/bash
#SBATCH --job-name=plotqa_check
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=plotqa_check_%j.out
#SBATCH --error=plotqa_check_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate vlmeval
cd /scratch/bh2863/fastv_project/VLMEvalKit

export LMUData=/scratch/bh2863/fastv_project/LMUData
mkdir -p $LMUData

python -c "
from vlmeval.dataset import build_dataset
d = build_dataset('PlotQA')
print('Total rows:', len(d.data))
print(d.data.columns.tolist())
print(d.data.head())
"
