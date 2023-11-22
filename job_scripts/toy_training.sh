#!/bin/bash
# apptainer exec \
#     --env PYTHONPATH=$(pwd) \
#     --env CUDA_VISIBLE_DEVICES=2,3 \
#     containers/c3se_job_container.sif \
#     python -m torch.distributed.launch --nproc_per_node=2 \
#     mmaction2/tools/train.py \
#     configs/models/videomaev2.py \
#     --launcher pytorch

apptainer exec \
    --env PYTHONPATH=$(pwd) \
    --env CUDA_VISIBLE_DEVICES=2,3 \
    containers/c3se_job_container.sif \
    python mmaction2/tools/train.py \
    configs/models/videomaev2.py