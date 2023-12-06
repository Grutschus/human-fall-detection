#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-1160 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --time=48:00:00

apptainer exec \
    --env PYTHONPATH=$(pwd) \
    containers/c3se_job_container.sif \
    python mmaction2/tools/train.py \
    configs/experiments/vit-b_frame-int-8_gaussian-sampling-5s-clips-30-drop_priority-labeling_k400-hyperparams.py