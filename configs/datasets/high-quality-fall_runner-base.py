"""Base `Runner` config for high-quality-fall dataset."""

dataset_type = "HighQualityFallDataset"

label_strategy = dict(
    type="PriorityLabel",
    label_description=dict(
        names=["fall", "lying", "other"],
        start_timestamp_names=["fall_start", "lying_start"],
        end_timestamp_names=["fall_end", "lying_end"],
        visible_names=["fall_visible", "lying_visible"],
        other_class=2,
    ),
)

sampling_strategy = dict(type="UniformSampling", clip_len=10)


# TRAIN
ann_file_train = "data/Fall_Simulation_Data/annotations_train.csv"

# TODO: Add shape comments
# TODO: Think about augmentation steps
train_pipeline = [
    dict(type="DecordInit"),
    dict(type="ClipVideo"),
    dict(type="SampleFrames", clip_len=16, frame_interval=4, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="RandomCrop", size=224),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=3,  # From VideoMAEv2 repo
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        sampling_strategy=sampling_strategy,
        label_strategy=label_strategy,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        num_classes=3,
        # indices=100,
    ),
)

# VALIDATION
ann_file_val = "data/Fall_Simulation_Data/annotations_val.csv"

val_pipeline = [
    dict(type="DecordInit"),
    dict(type="ClipVideo"),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=4, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="CenterCrop", crop_size=224),  # From VideoMAEv2 repo
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

# val_dataloader = train_dataloader
val_dataloader = dict(
    batch_size=3,  # From VideoMAEv2 repo
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        sampling_strategy=sampling_strategy,
        label_strategy=label_strategy,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        num_classes=3,
    ),
)

# TEST
ann_file_test = "data/Fall_Simulation_Data/annotations_test.csv"

test_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=4, num_clips=5, test_mode=True
    ),  # From VideoMAEv2 repo
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),  # From VideoMAEv2 repo
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

test_dataloader = dict(
    batch_size=3,  # From VideoMAEv2 repo
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        sampling_strategy=sampling_strategy,
        label_strategy=label_strategy,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        num_classes=3,
    ),
)
