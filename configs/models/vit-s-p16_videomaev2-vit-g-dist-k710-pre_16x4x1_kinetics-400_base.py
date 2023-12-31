_base_ = ["../default_runtime.py", "../datasets/high-quality-fall_runner-base.py"]

# Finetuning parameters are from VideoMAEv2 repo
# https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/FINETUNE.md


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
        drop_path_rate=0.3,  # From VideoMAEv2 repo
    ),
    cls_head=dict(
        type="TimeSformerHead",
        num_classes=3,
        in_channels=384,
        average_clips="prob",
        topk=(1,),
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
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=35, val_interval=1)

# TODO: Think about fine-tuning param scheduler
param_scheduler = [
    dict(
        type="LinearLR",
        by_epoch=True,
        convert_to_iter_based=True,
        start_factor=1e-3,
        end_factor=1,
        begin=0,
        end=5,
    ),  # From VideoMAEv2 repo - Warmup
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=1e-6,
        begin=5,
        end=35,
    ),
]

optim_wrapper = dict(
    type="AmpOptimWrapper",  # Automatic Mixed Precision may speed up trainig
    optimizer=dict(
        type="AdamW",  # From VideoMAEv2 repo
        lr=1e-3,  # From VideoMAEv2 repo
        weight_decay=0.1,  # From VideoMAEv2 repo
        betas=(0.9, 0.999),  # From VideoMAEv2 repo
    ),
    clip_grad=dict(max_norm=5, norm_type=2),  # From VideoMAEv2 repo
)

# VALIDATION CONFIG
val_evaluator = dict(
    type="AddAccMetric",
    metric_list=(
        "unweighted_average_f1",
        "per_class_f1",
        "per_class_precision",
        "per_class_recall",
    ),
)
val_cfg = dict(type="ValLoop")


# TEST CONFIG
test_evaluator = dict(
    type="AddAccMetric",
    metric_list=(
        "unweighted_average_f1",
        "per_class_f1",
        "per_class_precision",
        "per_class_recall",
    ),
)
test_cfg = dict(type="TestLoop")
