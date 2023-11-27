_base_ = ["../default_runtime.py", "../datasets/high-quality-fall_runner-base.py"]

# ViT-S-P16
model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="VisionTransformer",
        img_size=224,
        patch_size=16,
        embed_dims=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type="LN", eps=1e-6),
    ),
    cls_head=dict(
        type="TimeSformerHead",
        num_classes=3,
        in_channels=384,
        average_clips="prob",
        multi_class=True,
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[102.17311096191406, 98.78225708007812, 92.68714141845703],
        std=[58.04566192626953, 57.004024505615234, 57.3704948425293],
        format_shape="NCTHW",
    ),
)

# Loading weights
load_from = "weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth"

# TRAINING CONFIG
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=3)

# TODO: Think about fine-tuning param scheduler
param_scheduler = dict(
    type="MultiStepLR",  # Decays the learning rate once the number of epoch reaches one of the milestones
    begin=0,  # Step at which to start updating the learning rate
    end=100,  # Step at which to stop updating the learning rate
    by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
    milestones=[40, 80],  # Steps to decay the learning rate
    gamma=0.1,
)

optim_wrapper = dict(
    type="OptimWrapper",  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(  # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type="SGD",  # Name of optimizer
        lr=0.01,  # Learning rate
        momentum=0.9,  # Momentum factor
        weight_decay=0.0001,
    ),  # Weight decay
    clip_grad=dict(max_norm=40, norm_type=2),
)

# VALIDATION CONFIG
val_evaluator = dict(type="AccMetric")
val_cfg = dict(type="ValLoop")


# TEST CONFIG
test_evaluator = dict(type="AccMetric")
test_cfg = dict(type="TestLoop")
