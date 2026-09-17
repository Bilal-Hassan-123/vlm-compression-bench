#!/bin/bash
#SBATCH --job-name=chartqapro_fastv
#SBATCH -p preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -o chartqapro_infer_%j.out
#SBATCH -e chartqapro_infer_%j.err

source ~/.bashrc
conda activate fastv
cd /scratch/bh2863/fastv_project/VLMEvalKit
python infer_chartqapro_fastv.py
