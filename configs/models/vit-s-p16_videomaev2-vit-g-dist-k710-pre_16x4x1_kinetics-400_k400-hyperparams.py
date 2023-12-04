_base_ = [
    "../default_runtime.py",
    "../datasets/high-quality-fall_runner_k400-hyperparams.py",
]

# Finetuning parameters are from VideoMAEv2 repo
# https://github.com/OpenGVLab/VideoMAEv2/blob/master/scripts/finetune/vit_b_k400_ft.sh


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
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=90, val_interval=3)

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

# Layer Decay and Weight Decay module configs
vit_b_layer_decay_75_custom_keys = {
    "backbone.patch_embed.projection.weight": {
        "lr_mult": 0.023757264018058777,
        "decay_mult": 1,
    },
    "backbone.patch_embed.projection.bias": {
        "lr_mult": 0.023757264018058777,
        "decay_mult": 0,
    },
    "backbone.blocks.0.norm1.weight": {"lr_mult": 0.03167635202407837, "decay_mult": 0},
    "backbone.blocks.0.norm1.bias": {"lr_mult": 0.03167635202407837, "decay_mult": 0},
    "backbone.blocks.0.attn.q_bias": {"lr_mult": 0.03167635202407837, "decay_mult": 0},
    "backbone.blocks.0.attn.v_bias": {"lr_mult": 0.03167635202407837, "decay_mult": 0},
    "backbone.blocks.0.attn.proj.bias": {
        "lr_mult": 0.03167635202407837,
        "decay_mult": 0,
    },
    "backbone.blocks.0.norm2.weight": {"lr_mult": 0.03167635202407837, "decay_mult": 0},
    "backbone.blocks.0.norm2.bias": {"lr_mult": 0.03167635202407837, "decay_mult": 0},
    "backbone.blocks.0.mlp.layers.0.0.bias": {
        "lr_mult": 0.03167635202407837,
        "decay_mult": 0,
    },
    "backbone.blocks.0.mlp.layers.1.bias": {
        "lr_mult": 0.03167635202407837,
        "decay_mult": 0,
    },
    "backbone.blocks.0.attn.qkv.weight": {
        "lr_mult": 0.03167635202407837,
        "decay_mult": 1,
    },
    "backbone.blocks.0.attn.proj.weight": {
        "lr_mult": 0.03167635202407837,
        "decay_mult": 1,
    },
    "backbone.blocks.0.mlp.layers.0.0.weight": {
        "lr_mult": 0.03167635202407837,
        "decay_mult": 1,
    },
    "backbone.blocks.0.mlp.layers.1.weight": {
        "lr_mult": 0.03167635202407837,
        "decay_mult": 1,
    },
    "backbone.blocks.1.norm1.weight": {"lr_mult": 0.04223513603210449, "decay_mult": 0},
    "backbone.blocks.1.norm1.bias": {"lr_mult": 0.04223513603210449, "decay_mult": 0},
    "backbone.blocks.1.attn.q_bias": {"lr_mult": 0.04223513603210449, "decay_mult": 0},
    "backbone.blocks.1.attn.v_bias": {"lr_mult": 0.04223513603210449, "decay_mult": 0},
    "backbone.blocks.1.attn.proj.bias": {
        "lr_mult": 0.04223513603210449,
        "decay_mult": 0,
    },
    "backbone.blocks.1.norm2.weight": {"lr_mult": 0.04223513603210449, "decay_mult": 0},
    "backbone.blocks.1.norm2.bias": {"lr_mult": 0.04223513603210449, "decay_mult": 0},
    "backbone.blocks.1.mlp.layers.0.0.bias": {
        "lr_mult": 0.04223513603210449,
        "decay_mult": 0,
    },
    "backbone.blocks.1.mlp.layers.1.bias": {
        "lr_mult": 0.04223513603210449,
        "decay_mult": 0,
    },
    "backbone.blocks.1.attn.qkv.weight": {
        "lr_mult": 0.04223513603210449,
        "decay_mult": 1,
    },
    "backbone.blocks.1.attn.proj.weight": {
        "lr_mult": 0.04223513603210449,
        "decay_mult": 1,
    },
    "backbone.blocks.1.mlp.layers.0.0.weight": {
        "lr_mult": 0.04223513603210449,
        "decay_mult": 1,
    },
    "backbone.blocks.1.mlp.layers.1.weight": {
        "lr_mult": 0.04223513603210449,
        "decay_mult": 1,
    },
    "backbone.blocks.2.norm1.weight": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 0,
    },
    "backbone.blocks.2.norm1.bias": {"lr_mult": 0.056313514709472656, "decay_mult": 0},
    "backbone.blocks.2.attn.q_bias": {"lr_mult": 0.056313514709472656, "decay_mult": 0},
    "backbone.blocks.2.attn.v_bias": {"lr_mult": 0.056313514709472656, "decay_mult": 0},
    "backbone.blocks.2.attn.proj.bias": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 0,
    },
    "backbone.blocks.2.norm2.weight": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 0,
    },
    "backbone.blocks.2.norm2.bias": {"lr_mult": 0.056313514709472656, "decay_mult": 0},
    "backbone.blocks.2.mlp.layers.0.0.bias": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 0,
    },
    "backbone.blocks.2.mlp.layers.1.bias": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 0,
    },
    "backbone.blocks.2.attn.qkv.weight": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 1,
    },
    "backbone.blocks.2.attn.proj.weight": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 1,
    },
    "backbone.blocks.2.mlp.layers.0.0.weight": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 1,
    },
    "backbone.blocks.2.mlp.layers.1.weight": {
        "lr_mult": 0.056313514709472656,
        "decay_mult": 1,
    },
    "backbone.blocks.3.norm1.weight": {"lr_mult": 0.07508468627929688, "decay_mult": 0},
    "backbone.blocks.3.norm1.bias": {"lr_mult": 0.07508468627929688, "decay_mult": 0},
    "backbone.blocks.3.attn.q_bias": {"lr_mult": 0.07508468627929688, "decay_mult": 0},
    "backbone.blocks.3.attn.v_bias": {"lr_mult": 0.07508468627929688, "decay_mult": 0},
    "backbone.blocks.3.attn.proj.bias": {
        "lr_mult": 0.07508468627929688,
        "decay_mult": 0,
    },
    "backbone.blocks.3.norm2.weight": {"lr_mult": 0.07508468627929688, "decay_mult": 0},
    "backbone.blocks.3.norm2.bias": {"lr_mult": 0.07508468627929688, "decay_mult": 0},
    "backbone.blocks.3.mlp.layers.0.0.bias": {
        "lr_mult": 0.07508468627929688,
        "decay_mult": 0,
    },
    "backbone.blocks.3.mlp.layers.1.bias": {
        "lr_mult": 0.07508468627929688,
        "decay_mult": 0,
    },
    "backbone.blocks.3.attn.qkv.weight": {
        "lr_mult": 0.07508468627929688,
        "decay_mult": 1,
    },
    "backbone.blocks.3.attn.proj.weight": {
        "lr_mult": 0.07508468627929688,
        "decay_mult": 1,
    },
    "backbone.blocks.3.mlp.layers.0.0.weight": {
        "lr_mult": 0.07508468627929688,
        "decay_mult": 1,
    },
    "backbone.blocks.3.mlp.layers.1.weight": {
        "lr_mult": 0.07508468627929688,
        "decay_mult": 1,
    },
    "backbone.blocks.4.norm1.weight": {"lr_mult": 0.1001129150390625, "decay_mult": 0},
    "backbone.blocks.4.norm1.bias": {"lr_mult": 0.1001129150390625, "decay_mult": 0},
    "backbone.blocks.4.attn.q_bias": {"lr_mult": 0.1001129150390625, "decay_mult": 0},
    "backbone.blocks.4.attn.v_bias": {"lr_mult": 0.1001129150390625, "decay_mult": 0},
    "backbone.blocks.4.attn.proj.bias": {
        "lr_mult": 0.1001129150390625,
        "decay_mult": 0,
    },
    "backbone.blocks.4.norm2.weight": {"lr_mult": 0.1001129150390625, "decay_mult": 0},
    "backbone.blocks.4.norm2.bias": {"lr_mult": 0.1001129150390625, "decay_mult": 0},
    "backbone.blocks.4.mlp.layers.0.0.bias": {
        "lr_mult": 0.1001129150390625,
        "decay_mult": 0,
    },
    "backbone.blocks.4.mlp.layers.1.bias": {
        "lr_mult": 0.1001129150390625,
        "decay_mult": 0,
    },
    "backbone.blocks.4.attn.qkv.weight": {
        "lr_mult": 0.1001129150390625,
        "decay_mult": 1,
    },
    "backbone.blocks.4.attn.proj.weight": {
        "lr_mult": 0.1001129150390625,
        "decay_mult": 1,
    },
    "backbone.blocks.4.mlp.layers.0.0.weight": {
        "lr_mult": 0.1001129150390625,
        "decay_mult": 1,
    },
    "backbone.blocks.4.mlp.layers.1.weight": {
        "lr_mult": 0.1001129150390625,
        "decay_mult": 1,
    },
    "backbone.blocks.5.norm1.weight": {"lr_mult": 0.13348388671875, "decay_mult": 0},
    "backbone.blocks.5.norm1.bias": {"lr_mult": 0.13348388671875, "decay_mult": 0},
    "backbone.blocks.5.attn.q_bias": {"lr_mult": 0.13348388671875, "decay_mult": 0},
    "backbone.blocks.5.attn.v_bias": {"lr_mult": 0.13348388671875, "decay_mult": 0},
    "backbone.blocks.5.attn.proj.bias": {"lr_mult": 0.13348388671875, "decay_mult": 0},
    "backbone.blocks.5.norm2.weight": {"lr_mult": 0.13348388671875, "decay_mult": 0},
    "backbone.blocks.5.norm2.bias": {"lr_mult": 0.13348388671875, "decay_mult": 0},
    "backbone.blocks.5.mlp.layers.0.0.bias": {
        "lr_mult": 0.13348388671875,
        "decay_mult": 0,
    },
    "backbone.blocks.5.mlp.layers.1.bias": {
        "lr_mult": 0.13348388671875,
        "decay_mult": 0,
    },
    "backbone.blocks.5.attn.qkv.weight": {"lr_mult": 0.13348388671875, "decay_mult": 1},
    "backbone.blocks.5.attn.proj.weight": {
        "lr_mult": 0.13348388671875,
        "decay_mult": 1,
    },
    "backbone.blocks.5.mlp.layers.0.0.weight": {
        "lr_mult": 0.13348388671875,
        "decay_mult": 1,
    },
    "backbone.blocks.5.mlp.layers.1.weight": {
        "lr_mult": 0.13348388671875,
        "decay_mult": 1,
    },
    "backbone.blocks.6.norm1.weight": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.norm1.bias": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.attn.q_bias": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.attn.v_bias": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.attn.proj.bias": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.norm2.weight": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.norm2.bias": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.mlp.layers.0.0.bias": {
        "lr_mult": 0.177978515625,
        "decay_mult": 0,
    },
    "backbone.blocks.6.mlp.layers.1.bias": {"lr_mult": 0.177978515625, "decay_mult": 0},
    "backbone.blocks.6.attn.qkv.weight": {"lr_mult": 0.177978515625, "decay_mult": 1},
    "backbone.blocks.6.attn.proj.weight": {"lr_mult": 0.177978515625, "decay_mult": 1},
    "backbone.blocks.6.mlp.layers.0.0.weight": {
        "lr_mult": 0.177978515625,
        "decay_mult": 1,
    },
    "backbone.blocks.6.mlp.layers.1.weight": {
        "lr_mult": 0.177978515625,
        "decay_mult": 1,
    },
    "backbone.blocks.7.norm1.weight": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.norm1.bias": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.attn.q_bias": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.attn.v_bias": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.attn.proj.bias": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.norm2.weight": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.norm2.bias": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.mlp.layers.0.0.bias": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.mlp.layers.1.bias": {"lr_mult": 0.2373046875, "decay_mult": 0},
    "backbone.blocks.7.attn.qkv.weight": {"lr_mult": 0.2373046875, "decay_mult": 1},
    "backbone.blocks.7.attn.proj.weight": {"lr_mult": 0.2373046875, "decay_mult": 1},
    "backbone.blocks.7.mlp.layers.0.0.weight": {
        "lr_mult": 0.2373046875,
        "decay_mult": 1,
    },
    "backbone.blocks.7.mlp.layers.1.weight": {"lr_mult": 0.2373046875, "decay_mult": 1},
    "backbone.blocks.8.norm1.weight": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.norm1.bias": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.attn.q_bias": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.attn.v_bias": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.attn.proj.bias": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.norm2.weight": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.norm2.bias": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.mlp.layers.0.0.bias": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.mlp.layers.1.bias": {"lr_mult": 0.31640625, "decay_mult": 0},
    "backbone.blocks.8.attn.qkv.weight": {"lr_mult": 0.31640625, "decay_mult": 1},
    "backbone.blocks.8.attn.proj.weight": {"lr_mult": 0.31640625, "decay_mult": 1},
    "backbone.blocks.8.mlp.layers.0.0.weight": {"lr_mult": 0.31640625, "decay_mult": 1},
    "backbone.blocks.8.mlp.layers.1.weight": {"lr_mult": 0.31640625, "decay_mult": 1},
    "backbone.blocks.9.norm1.weight": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.norm1.bias": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.attn.q_bias": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.attn.v_bias": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.attn.proj.bias": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.norm2.weight": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.norm2.bias": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.mlp.layers.0.0.bias": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.mlp.layers.1.bias": {"lr_mult": 0.421875, "decay_mult": 0},
    "backbone.blocks.9.attn.qkv.weight": {"lr_mult": 0.421875, "decay_mult": 1},
    "backbone.blocks.9.attn.proj.weight": {"lr_mult": 0.421875, "decay_mult": 1},
    "backbone.blocks.9.mlp.layers.0.0.weight": {"lr_mult": 0.421875, "decay_mult": 1},
    "backbone.blocks.9.mlp.layers.1.weight": {"lr_mult": 0.421875, "decay_mult": 1},
    "backbone.blocks.10.norm1.weight": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.norm1.bias": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.attn.q_bias": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.attn.v_bias": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.attn.proj.bias": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.norm2.weight": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.norm2.bias": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.mlp.layers.0.0.bias": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.mlp.layers.1.bias": {"lr_mult": 0.5625, "decay_mult": 0},
    "backbone.blocks.10.attn.qkv.weight": {"lr_mult": 0.5625, "decay_mult": 1},
    "backbone.blocks.10.attn.proj.weight": {"lr_mult": 0.5625, "decay_mult": 1},
    "backbone.blocks.10.mlp.layers.0.0.weight": {"lr_mult": 0.5625, "decay_mult": 1},
    "backbone.blocks.10.mlp.layers.1.weight": {"lr_mult": 0.5625, "decay_mult": 1},
    "backbone.blocks.11.norm1.weight": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.norm1.bias": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.attn.q_bias": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.attn.v_bias": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.attn.proj.bias": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.norm2.weight": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.norm2.bias": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.mlp.layers.0.0.bias": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.mlp.layers.1.bias": {"lr_mult": 0.75, "decay_mult": 0},
    "backbone.blocks.11.attn.qkv.weight": {"lr_mult": 0.75, "decay_mult": 1},
    "backbone.blocks.11.attn.proj.weight": {"lr_mult": 0.75, "decay_mult": 1},
    "backbone.blocks.11.mlp.layers.0.0.weight": {"lr_mult": 0.75, "decay_mult": 1},
    "backbone.blocks.11.mlp.layers.1.weight": {"lr_mult": 0.75, "decay_mult": 1},
    "backbone.fc_norm.weight": {"lr_mult": 1.0, "decay_mult": 0},
    "backbone.fc_norm.bias": {"lr_mult": 1.0, "decay_mult": 0},
    "cls_head.fc_cls.bias": {"lr_mult": 1.0, "decay_mult": 0},
    "cls_head.fc_cls.weight": {"lr_mult": 1.0, "decay_mult": 1},
}


optim_wrapper = dict(
    type="AmpOptimWrapper",  # Automatic Mixed Precision may speed up trainig
    optimizer=dict(
        type="AdamW",  # From VideoMAEv2 repo
        lr=7e-4,  # From VideoMAEv2 repo
        weight_decay=0.05,  # From VideoMAEv2 repo
        betas=(0.9, 0.999),  # From VideoMAEv2 repo
    ),
    paramwise_cfg=dict(custom_keys=vit_b_layer_decay_75_custom_keys),
    # clip_grad=dict(max_norm=5, norm_type=2),  # From VideoMAEv2 repo
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
