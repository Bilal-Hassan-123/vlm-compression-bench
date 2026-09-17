#!/bin/bash
#SBATCH --job-name=chartbench_check
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=chartbench_check_%j.out
#SBATCH --error=chartbench_check_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate vlmeval
cd /scratch/bh2863/fastv_project/VLMEvalKit

python -c "
from vlmeval.dataset import build_dataset
d = build_dataset('ChartBench')
print('Total rows:', len(d.data))
print(d.data.columns.tolist())
print(d.data[['index','question','answer']].head(6).to_string())
print(d.data['QA_type'].value_counts() if 'QA_type' in d.data.columns else 'no QA_type col')
"
