#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-1160 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4
#SBATCH --time=24:00:00

apptainer exec \
    --env PYTHONPATH=$(pwd) \
    containers/c3se_job_container.sif \
    python -m torch.distributed.launch --nproc_per_node=4 \
    mmaction2/tools/test.py \
    configs/experiments/vit-b_frame-int-8_cutup-5s-clips-30-drop_priority-labeling_k400-hyperparams.py \
    experiments/frame-int-8_cutup-5s-clips-30-drop_fixed_lr/best_acc_unweighted_average_f1_epoch_30.pth \
    --work-dir model_tests/high_temporal_resolution_frame-int-8_cutup-5s-clips-30-drop_fixed_lr \
    --dump model_tests/high_temporal_resolution_frame-int-8_cutup-5s-clips-30-drop_fixed_lr/predictions.pkl \
    --show-dir model_tests/high_temporal_resolution_frame-int-8_cutup-5s-clips-30-drop_fixed_lr/visualizations \
    --interval 2000 \
    --launcher pytorch