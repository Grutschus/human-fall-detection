_base_ = [
    "../models/vit-s-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_k400-hyperparams.py"
]

EXPERIMENT_NAME = "vit-b_gaussian-sampling_priority-labeling_k400-hyperparams"
visualizer = dict(
    vis_backends=dict(save_dir=f"experiments/tensorboard/{EXPERIMENT_NAME}/")
)
work_dir = f"experiments/{EXPERIMENT_NAME}"

# Overrides
default_hooks = dict(checkpoint=dict(interval=1))

# 1487 samples in val -> 92 batches per node -> We want around 10 images
custom_hooks = [dict(type="CustomVisualizationHook", enable=True, interval=300)]

# Use ViT-B/16
model = dict(
    backbone=dict(embed_dims=768, depth=12, num_heads=12),
    cls_head=dict(in_channels=768),
)
load_from = "weights/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"

# Use Gaussian sampling
train_dataloader = dict(
    dataset=dict(sampling_strategy=dict(type="GaussianSampling", clip_len=10))
)
# We are not changing the val/test dataloaders since gaussian sampling requires labels
# and we cannot have a valid validation if we use labels in the preprocessing
