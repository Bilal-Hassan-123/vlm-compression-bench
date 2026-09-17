#!/bin/bash
#SBATCH --job-name=eval_visionzip_llavanext
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=eval_visionzip_llavanext_%j.log
#SBATCH --error=eval_visionzip_llavanext_%j.err

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/bh2863/conda/envs/vlmeval
which python
python --version

export HF_HOME=/scratch/bh2863/HF_HOME
export LMUData=/scratch/bh2863/LMUData
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd /scratch/bh2863/fastv_project/VLMEvalKit
python run.py --data ChartQA_TEST --model visionzip_llavanext_7b --mode eval
