_base_ = [
    "../models/vit-s-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_k400-hyperparams.py"
]

EXPERIMENT_NAME = "high_temporal_resolution_frame-int-8_cutup-5s-clips-30-drop_fixed_lr"
visualizer = dict(
    vis_backends=dict(save_dir=f"model_tests/tensorboard/{EXPERIMENT_NAME}")
)
work_dir = f"model_tests/{EXPERIMENT_NAME}"

# Overrides
default_hooks = dict(checkpoint=dict(interval=1))

# 1487 samples in val -> 92 batches per node -> We want around 10 images
# custom_hooks = [dict(type="CustomVisualizationHook", enable=True, interval=300)]

# Use ViT-B/16
model = dict(
    backbone=dict(embed_dims=768, depth=12, num_heads=12),
    cls_head=dict(in_channels=768),
)
load_from = "weights/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"

# Use frame_interval 8
train_pipeline = [
    dict(type="DecordInit"),
    dict(type="ClipVideo"),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=8, num_clips=1
    ),  # This has changed
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="RandomCrop", size=224),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]


# Use Cutup sampling
train_dataloader = dict(
    dataset=dict(
        sampling_strategy=dict(
            type="UniformSampling",
            clip_len=5,
        ),
        drop_ratios=[0.0, 0.0, 0.30],
        pipeline=train_pipeline,
    )
)
# We are not changing the val/test dataloaders since gaussian sampling requires labels
# and we cannot have a valid validation if we use labels in the preprocessing

val_pipeline = [
    dict(type="DecordInit"),
    dict(type="ClipVideo"),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=8, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="CenterCrop", crop_size=224),  # From VideoMAEv2 repo
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

val_dataloader = dict(
    dataset=dict(
        sampling_strategy=dict(
            type="UniformSampling", clip_len=5, stride=0, overlap=False
        ),
        pipeline=val_pipeline,
    ),
)


test_pipeline = [
    dict(type="DecordInit"),
    dict(type="ClipVideo"),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=8, num_clips=5, test_mode=True
    ),  # From VideoMAEv2 repo
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),  # From VideoMAEv2 repo
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

test_dataloader = dict(
    num_workers=2,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        pipeline=test_pipeline,
        sampling_strategy=dict(
            type="UniformSampling", clip_len=5, stride=1, overlap=True
        ),
    ),
)
