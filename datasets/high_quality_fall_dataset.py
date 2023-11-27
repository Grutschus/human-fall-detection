from typing import Callable, List, Optional, Union

import pandas as pd
from mmaction.datasets import BaseActionDataset
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmengine.fileio import exists

from datasets.transforms.label_strategy import LabelStrategy
from datasets.transforms.sampling_strategy import SamplingStrategy
from registry import LABEL_STRATEGIES, SAMPLING_STRATEGIES


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
        - length: length of the video in seconds
        - fall_visible: boolean indicating whether the fall is visible
            on the video
        - lying_visible: boolean indicating whether the lying is visible
            on the video

    Example of a annotation file:

    | video_path                                       | fall_start | fall_end | lying_start | lying_end | length |fall_visible | lying_visible |
    |--------------------------------------------------|------------|----------|-------------|-----------|--------|-------------|---------------|
    | data/Fall_Simulation_Data/videos/Fall30_Cam3.avi | 24.0       | 27.0     | 27.0        | 88.0      | 240.0  |True         | True          |
    | data/Fall_Simulation_Data/videos/ADL16_Cam1.avi  |            |          |             |           | 325    |             |               |

    Args:
        ann_file (str): Path to the annotation file.
        sampling_strategy (SamplingStrategy): Strategy used to sample clips.
        label_strategy (LabelStrategy): Strategy used to label clips.
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
        sampling_strategy: SamplingStrategy | dict,
        label_strategy: LabelStrategy | dict,
        pipeline: List[Union[dict, Callable]],
        data_prefix: ConfigType = dict(video=""),
        multi_class: bool = False,
        num_classes: Optional[int] = None,
        start_index: int = 0,
        modality: str = "RGB",
        test_mode: bool = False,
        **kwargs,
    ) -> None:
        # Bug in MMENGINE: kwarg `custom_imports` is not removed from kwargs
        # this causes an error when building the dataset
        # TODO: Create an issue on MMENGINE, can be fixed here:
        # https://github.com/open-mmlab/mmengine/blob/85c0976bc2434157f786d44cdd8f0fb2955414f0/mmengine/config/config.py#L462C34-L462C34
        kwargs.pop("custom_imports", None)

        if isinstance(sampling_strategy, dict):
            self.sampling_strategy = SAMPLING_STRATEGIES.build(sampling_strategy)  # type: SamplingStrategy
        else:
            self.sampling_strategy = sampling_strategy
        if isinstance(label_strategy, dict):
            self.label_strategy = LABEL_STRATEGIES.build(label_strategy)  # type: LabelStrategy
        else:
            self.label_strategy = label_strategy
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
        exists(self.ann_file)
        annotations = pd.read_csv(self.ann_file)
        data_list = []
        for _, annotation in annotations.iterrows():
            sampled_clips = self.sampling_strategy.sample(annotation)
            labels = [
                self.label_strategy.label(annotation, clip) for clip in sampled_clips
            ]

            for clip, label in zip(sampled_clips, labels):
                data_list.append(
                    {
                        "filename": annotation["video_path"],
                        "label": label,
                        "interval": clip,
                    }
                )
        return data_list
