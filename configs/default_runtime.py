"""Default runtime for our experiments."""

# Trying to skip this part, since we have custom registries not in this scope
default_scope = "mmaction"
work_dir = "experiments"
custom_imports = dict(
    imports=["datasets", "evaluation", "visualization"], allow_failed_imports=False
)
launcher = "none"

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=3,
        by_epoch=True,
        max_keep_ckpts=3,
        save_best="auto",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
    visualization=dict(type="CustomVisualizationHook"),
)

# Hook disabled since it cannot handle NCTHW tensors
# TODO fix this
# custom_hooks = [dict(type="CustomVisualizationHook", enable=True)]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(
    type="LogProcessor",
    window_size=10,
    by_epoch=True,
)

vis_backends = dict(type="TensorboardVisBackend", save_dir="experiments/tensorboard")
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)

log_level = "INFO"

# Overwrite this to continue training
load_from = None
resume = False
