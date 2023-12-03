_base_ = [
    "../models/vit-s-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_base.py"
]

EXPERIMENT_NAME = "vit-b_gaussian-sampling_priority-labeling_paper-hyperparams"
visualizer = dict(
    vis_backends=dict(save_dir=f"experiments/tensorboard/{EXPERIMENT_NAME}/")
)
work_dir = f"experiments/{EXPERIMENT_NAME}"

# Overrides
default_hooks = dict(checkpoint=dict(interval=3))

# 1487 samples in val -> 372 per node -> 124 batches per node -> We want around 10 images
# -> Interval = 124 / 10 = 12
custom_hooks = [dict(type="CustomVisualizationHook", enable=True, interval=10)]

# Use ViT-B/16
# Add weighted CE loss
# weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
model = dict(
    backbone=dict(embed_dims=768, depth=12, num_heads=12),
    cls_head=dict(
        in_channels=768,
        loss_cls=dict(
            type="CrossEntropyLoss",
            class_weight=[26.38235294117647, 37.901408450704224, 3.7168508287292816],
        ),
    ),
)
load_from = "weights/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"

# Use Gaussian sampling
train_dataloader = dict(
    dataset=dict(sampling_strategy=dict(type="GaussianSampling", clip_len=10))
)
# We are not changing the val/test dataloaders since gaussian sampling requires labels
# and we cannot have a valid validation if we use labels in the preprocessing
