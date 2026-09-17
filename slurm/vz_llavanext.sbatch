#!/bin/bash
#SBATCH --job-name=infer_vz_next
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=infer_vz_next_%j.out

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate visionzip

export HF_HOME=/scratch/bh2863/HF_HOME
export PYTHONPATH=/scratch/bh2863/fastv_project/clean-LLaVA:/scratch/bh2863/fastv_project/VisionZip:$PYTHONPATH

cd /scratch/bh2863/fastv_project/VLMEvalKit    # CSV uses relative image paths
python infer_chartqa_llavanext_visionzip.py
