_base_ = ["../../mmaction2/configs/_base_/default_runtime.py"]

custom_imports = dict(imports="datasets", allow_failed_imports=False)
work_dir = "work_dirs/videomaev2"
launcher = "none"


# model settings
model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="VisionTransformer",
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type="LN", eps=1e-6),
    ),
    cls_head=dict(
        type="TimeSformerHead", num_classes=3, in_channels=768, average_clips="prob"
    ),
    # TODO: update this to fit our dataset
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape="NCTHW",
    ),
)

# dataset settings
dataset_type = "HighQualityFallDataset"
ann_file_train = "tests/test_data/test_annotation.csv"

train_pipeline = [
    dict(type="DecordInit"),
    dict(type="ClipVideo"),
    dict(type="SampleFrames", clip_len=16, frame_interval=4, num_clips=5),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        sampling_strategy=dict(type="UniformSampling", clip_len=10),
        label_strategy=dict(
            type="ExistenceLabel",
            label_description=dict(
                names=["fall", "lying", "other"],
                start_timestamp_names=["fall_start", "lying_start"],
                end_timestamp_names=["fall_end", "lying_end"],
                visible_names=["fall_visible", "lying_visible"],
                other_class=2,
            ),
        ),
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        multiclass=True,
        num_classes=3,
    ),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=5, val_interval=0)

param_scheduler = dict(
    type="MultiStepLR",  # Decays the learning rate once the number of epoch reaches one of the milestones
    begin=0,  # Step at which to start updating the learning rate
    end=100,  # Step at which to stop updating the learning rate
    by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
    milestones=[40, 80],  # Steps to decay the learning rate
    gamma=0.1,
)

optim_wrapper = dict(  # Config of optimizer wrapper
    type="OptimWrapper",  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(  # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type="SGD",  # Name of optimizer
        lr=0.01,  # Learning rate
        momentum=0.9,  # Momentum factor
        weight_decay=0.0001,
    ),  # Weight decay
    clip_grad=dict(max_norm=40, norm_type=2),
)  # Config of gradient clip
