from typing import Callable, List, Optional, Union

from mmaction.datasets import BaseActionDataset  # type: ignore
from mmaction.registry import DATASETS  # type: ignore
from mmaction.utils import ConfigType


@DATASETS.register_module()
class HighQualityFallDataset(BaseActionDataset):
    """HighQualityFallDataset dataset for action recognition.

    The dataset loads raw videos and applies specified transforms to return
    a dict containing the frame tensors and other information.

    It will sample clips from longer videos according to a `SamplingStrategy`
    and add an `interval` key to the resulting dict.
    This key may be processed by the `ClipVideo` transform.

    The ann_file is a CSV file with the following columns:
        - video_path: relative path to the video from the data prefix
        - fall_start: timestamp in seconds of the start of the fall
        - fall_end: timestamp in seconds of the end of the fall
        - lying_start: timestamp in seconds of the start of the lying
        - lying_end: timestamp in seconds of the end of the lying
        - fall_visible: boolean indicating whether the fall is visible
            on the video
        - lying_visible: boolean indicating whether the lying is visible
            on the video

    Example of a annotation file:

    | video_path                                       | fall_start | fall_end | lying_start | lying_end | fall_visible | lying_visible |
    |--------------------------------------------------|------------|----------|-------------|-----------|--------------|---------------|
    | data/Fall_Simulation_Data/videos/Fall30_Cam3.avi | 24.0       | 27.0     | 27.0        | 88.0      | True         | True          |
    | data/Fall_Simulation_Data/videos/ADL16_Cam1.avi  |            |          |             |           |              |               |

    Args:
        ann_file (str): Path to the annotation file.
        sampling_strategy (SamplingStrategy): Callable used to sample.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms. Should include a `ClipVideo` transform.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``'RGB'``, ``'Flow'``.
            Defaults to ``'RGB'``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False."""

    def __init__(
        self,
        ann_file: str,
        sampling_strategy: Callable,  # TODO: create sampling strategy type
        pipeline: List[Union[dict, Callable]],
        data_prefix: ConfigType = dict(video=""),
        multi_class: bool = False,
        num_classes: Optional[int] = None,
        start_index: int = 0,
        modality: str = "RGB",
        test_mode: bool = False,
        **kwargs,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=test_mode,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        return super().load_data_list()

    def get_data_info(self, idx: int) -> dict:
        return super().get_data_info(idx)
