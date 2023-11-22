import abc
from typing import List, Tuple

import pandas as pd

from registry import SAMPLING_STRATEGIES

IntervalInSeconds = Tuple[float, float]


class SamplingStrategy(abc.ABC):
    """Generic sampling strategy. Used to sample clips as intervals from
    untrimmed videos."""

    @abc.abstractmethod
    def sample(self, annotation: pd.Series) -> List[IntervalInSeconds]:
        """Samples intervals from the untrimmed video.

        Args:
            annotation (pd.Series): Annotation of an untrimmed video. The
                annotation has to adhere to the same format as the `HighQualityFallDataset`
                annotation file.

        Returns:
            List[IntervalInSeconds]: List of samples as intervals in seconds."""
        ...


@SAMPLING_STRATEGIES.register_module()
class UniformSampling(SamplingStrategy):
    """Samples uniformly from the untrimmed video.

    For a video of length `L` and a clip length `C`, the video is split into
    `floor(L/C)` clips of length `C`. The beginning of each next samples is offset
    by the `stride`.

    Example: For a video of length 10s, a clip length of 2s and a stride of 1s,
    the first two intervals are (0, 2) and (3, 5).

    Args:
        clip_len (float): Length of the clips to sample in seconds.
        stride (float): Stride between the clips in seconds. Defaults to 0.
        overlap (bool): Whether to overlap clips. Stride has to be larger than
            0 in this case. Defaults to False.
    """

    def __init__(
        self, clip_len: float, stride: float = 0.0, overlap: bool = False
    ) -> None:
        self.clip_len = clip_len
        self.stride = stride
        self.overlap = overlap

    def sample(self, annotation: pd.Series) -> List[IntervalInSeconds]:
        total_length = annotation["length"]
        sample_list = []

        if self.overlap and self.stride <= 0:
            raise ValueError("Stride has to be larger than 0 when overlapping clips.")

        clip_start = 0.0
        while clip_start + self.clip_len <= total_length:
            sample_list.append((clip_start, clip_start + self.clip_len))
            clip_start += self.stride if self.overlap else self.clip_len + self.stride

        return sample_list
