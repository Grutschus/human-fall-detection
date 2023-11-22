from mmengine.registry import Registry


SAMPLING_STRATEGIES = Registry(
    "sampling strategy", locations=["datasets.transforms.sampling_strategy"], scope="."
)

LABEL_STRATEGIES = Registry(
    "label strategy", locations=["datasets.transforms.label_strategy"], scope="."
)
