#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-1160 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH --time=24:00:00

apptainer exec \
    --env PYTHONPATH=$(pwd) \
    containers/c3se_job_container.sif \
    python -m torch.distributed.launch --nproc_per_node=4 \
    mmaction2/tools/train.py \
    configs/experiments/vit-b_gaussian-sampling_priority-labeling_paper-hyperparams.py \
    --launcher pytorch --resume auto