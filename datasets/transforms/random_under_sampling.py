from mmaction.registry import TRANSFORMS  # type: ignore
from mmcv.transforms import BaseTransform  # type: ignore


@TRANSFORMS.register_module()
class RandomUnderSampling(BaseTransform):
    """Randomly drop samples from the given class."""

    def __init__(self, drop_ratio: float = 0.2) -> None:
        self.drop_ratio = drop_ratio

    def transform(self, results: dict) -> dict:
        """Perform the `RandomUnderSampling` transformation.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        raise NotImplementedError
