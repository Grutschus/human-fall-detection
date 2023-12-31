_base_ = [
    "../models/vit-s-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_k400-hyperparams.py"
]

EXPERIMENT_NAME = "overfitting_run"
visualizer = dict(
    vis_backends=dict(save_dir=f"experiments/tensorboard/{EXPERIMENT_NAME}/")
)
work_dir = f"experiments/{EXPERIMENT_NAME}"

# Overrides
train_dataloader = dict(
    batch_size=1,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        indices=100,
    ),
)

ann_file_val = "data/Fall_Simulation_Data/annotations_train.csv"

val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    batch_size=1,
    dataset=dict(
        ann_file=ann_file_val,
        indices=100,
    ),
)

default_hooks = dict(checkpoint=dict(interval=0))
custom_hooks = [dict(type="CustomVisualizationHook", enable=True, interval=1)]
