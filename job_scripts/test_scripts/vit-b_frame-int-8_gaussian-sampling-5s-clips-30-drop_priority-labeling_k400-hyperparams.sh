#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-1160 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --time=01:00:00

apptainer exec \
    --env PYTHONPATH=$(pwd) \
    containers/c3se_job_container.sif \
    python mmaction2/tools/test.py \
    configs/experiments/vit-b_frame-int-8_gaussian-sampling-5s-clips-30-drop_priority-labeling_k400-hyperparams.py \
    experiments/vit-b_frame-int-8_gaussian-sampling-5s-clips-30-drop_priority-labeling_k400-hyperparams/best_acc_unweighted_average_f1_epoch_16.pth \
    --work-dir model_tests/vit-b_frame-int-8_gaussian-sampling-5s-clips-30-drop_priority-labeling_k400-hyperparams \
    --dump model_tests/vit-b_frame-int-8_gaussian-sampling-5s-clips-30-drop_priority-labeling_k400-hyperparams/predictions.pkl \
    --show-dir model_tests/vit-b_frame-int-8_gaussian-sampling-5s-clips-30-drop_priority-labeling_k400-hyperparams/visualizations