#!/bin/bash
#SBATCH --job-name=cdpruner_llavanext
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=cdpruner_llavanext_%j.log
#SBATCH --error=cdpruner_llavanext_%j.err
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate cdpruner
export HF_HOME = /scratch/bh2863/HF_HOME
export LMUData = /scratch/bh2863/LMUData
cd /scratch/bh2863/fastv_project/VLMEvalKit
python infer_chartqa_llavanext_cdpruner.py

