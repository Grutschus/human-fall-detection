_base_ = [
    "../models/vit-s-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_base.py"
]

train_dataloader = dict(
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        indices=100,
    ),
)

ann_file_val = "data/Fall_Simulation_Data/annotations_train.csv"

val_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        indices=100,
    ),
)

default_hooks = dict(checkpoint=dict(interval=0))
custom_hooks = [dict(type="CustomVisualizationHook", enable=True, interval=10)]
