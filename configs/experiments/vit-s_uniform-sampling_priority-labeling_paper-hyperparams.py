_base_ = [
    "../models/vit-s-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_base.py"
]

EXPERIMENT_NAME = "vit-s_uniform-sampling_priority-labeling_paper-hyperparams"
visualizer = dict(
    vis_backends=dict(save_dir=f"experiments/tensorboard/{EXPERIMENT_NAME}/")
)
work_dir = f"experiments/{EXPERIMENT_NAME}"

# Overrides
default_hooks = dict(checkpoint=dict(interval=3))
# Roughly 2800 samples in eval -> 700 per node -> We get 10 images from the master with interval=70
custom_hooks = [dict(type="CustomVisualizationHook", enable=True, interval=70)]
