import abc
from typing import List, Tuple

import numpy as np
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


@SAMPLING_STRATEGIES.register_module()
class GaussianSampling(SamplingStrategy):
    """Samples the center of each interval from a gaussian distribution that
    is centered around the center of a given class interval.

    Example: If the priority class is `fall` and the fall interval is (2, 4),
        the center of clips are sampled from a Gaussian distribution centered around
        3. The standard deviation can be freely chosen.

    Args:
        clip_len (float): Length of the clips to sample in seconds.
        focus_interval_start_name (str): Name of the column in the annotation
            that contains the start timestamp of the priority class interval.
            Defaults to "fall_start".
        focus_interval_end_name (str): Name of the column in the annotation
            that contains the end timestamp of the priority class interval.
            Defaults to "fall_end".
        n_samples_per_sec (float | None): Number of samples per second. If None `1/clip_len`
            is used to sample approximately the same number of samples as a
            UniformSampling strategy. Defaults to None.
        fallback_sampler (SamplingStrategy | dict | None): Sampler to use if the timestamps
            of the focus interval are not present in the annotation. If None,
            `UniformSampling` with equal `clip_len` is used. Defaults to None.
        std (None | float): Standard deviation of the gaussian distribution. If None,
            `min(focus_interval_center, total_length - focus_interval_center) / 3` is used.
            Defaults to None.
    """

    def __init__(
        self,
        clip_len: float,
        focus_interval_start_name: str = "fall_start",
        focus_interval_end_name: str = "fall_end",
        n_samples_per_sec: float | None = None,
        fallback_sampler: SamplingStrategy | dict | None = None,
        std: None | float = None,
    ) -> None:
        self.clip_len = clip_len
        self.focus_interval_start_name = focus_interval_start_name
        self.focus_interval_end_name = focus_interval_end_name
        self.std = std
        if fallback_sampler is None:
            self.fallback_sampler: SamplingStrategy = UniformSampling(clip_len)
        elif isinstance(fallback_sampler, dict):
            self.fallback_sampler = SAMPLING_STRATEGIES.build(fallback_sampler)
        else:
            self.fallback_sampler = fallback_sampler
        self.n_samples_per_sec = n_samples_per_sec

    def sample(self, annotation: pd.Series) -> List[IntervalInSeconds]:
        if (
            self.focus_interval_start_name not in annotation.keys()
            or self.focus_interval_end_name not in annotation.keys()
        ):
            raise ValueError(
                "Given focus interval names "
                f"{self.focus_interval_start_name} and {self.focus_interval_end_name} "
                "are not in the annotation."
            )

        focus_interval = (
            annotation[self.focus_interval_start_name],
            annotation[self.focus_interval_end_name],
        )

        if any(np.isnan(focus_interval)):
            return self.fallback_sampler.sample(annotation)

        mean = sum(focus_interval) / 2
        std = self.std
        if std is None:
            std = min(mean, annotation["length"] - mean) / 3

        n_samples_per_sec = self.n_samples_per_sec
        if n_samples_per_sec is None:
            n_samples_per_sec = 1.0 / self.clip_len

        samples = np.random.normal(
            mean, std, int(n_samples_per_sec * annotation["length"])
        ).round(decimals=2)

        sample_list = []
        for sample in samples:
            start = max(0, sample - self.clip_len / 2)
            end = min(annotation["length"], sample + self.clip_len / 2)
            sample_list.append((start, end))

        return sample_list


@SAMPLING_STRATEGIES.register_module()
class FilterSampling(SamplingStrategy):
    """Meta-sampling strategy that performs video-level filtering.

    It drops complete videos from the dataset."""

    def __init__(
        self,
        sampler: SamplingStrategy | dict,
        filter_column_name: str = "category",
        values: str | list[str] | None = "ADL",
        blacklist: bool = True,
    ) -> None:
        if isinstance(sampler, dict):
            self.sampler = SAMPLING_STRATEGIES.build(sampler)
        else:
            self.sampler = sampler

    def sample(self, annotation: pd.Series) -> List[IntervalInSeconds]:
        raise NotImplementedError
        # Check whether the filter applies and we should discard the sample -> return empty list

        # Otherwise return the samples of the sampler
