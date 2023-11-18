import abc
from typing import List

import pandas as pd

from datasets.transforms.SamplingStrategy import IntervalInSeconds
from dataclasses import dataclass


@dataclass
class LabelDescription:
    """Dataclass for the description of the labels and their corresponding timestamps
        in the annotation file.

    Args:
        names (tuple[str]): Names of the actions.
        start_timestamp_names (tuple[str]): Names of the columns for the start timestamps.
        end_timestamp_names (tuple[str]): Names of the columns for the end timestamps.
        visible_names (tuple[str]): Names of the columns for the visibility of actions.
        other_class (int): Index of the `other` class.

    """

    names: tuple[str]
    start_timestamp_names: tuple[str]
    end_timestamp_names: tuple[str]
    visible_names: tuple[str]
    other_class: int


class LabelStrategy(abc.ABC):
    """Generic labeling strategy. Used to extract labels from an annotation
    and a given clip.

    Args:
        label_description (LabelDescription): Description of the labels."""

    def __init__(self, label_description: LabelDescription) -> None:
        self.label_description = label_description

    @abc.abstractmethod
    def label(self, annotation: pd.Series, clip: IntervalInSeconds) -> List[int]:
        """Extracts the label from the annotation and the clip.

        Args:
            annotation (pd.Series): Annotation of an untrimmed video. The
                annotation has to adhere to the same format as the `HighQualityFallDataset`
                annotation file.
            clip (IntervalInSeconds): Interval of the clip in seconds.

        Returns:
            List[int]: Labels of the clip."""
        ...


class ExistanceLabel(LabelStrategy):
    """Generates a label based on the existance of actions within the clip.

    If the clip contains an action, the corresponding label is given.

    Args:
        threshold (float): Threshold for the existance of an action. Defaults to 0.
        absolute_threshold (bool): Whether to use the threshold as an absolute value (True) in seconds
            or as a percentage of the clip length (False). Defaults to True
    """

    def __init__(self, threshold: float = 0.0, absolute_threshold: bool = True) -> None:
        self.threshold = threshold
        self.absolute_threshold = absolute_threshold

    def label(self, annotation: pd.Series, clip: IntervalInSeconds) -> List[int]:
        labels = []
        labeled_time = 0.0
        for idx, (start_name, end_name, visible_name) in enumerate(
            zip(
                self.label_description.start_timestamp_names,
                self.label_description.end_timestamp_names,
                self.label_description.visible_names,
            )
        ):
            # We take care of the `other` class later
            if idx == self.label_description.other_class:
                continue

            if annotation[visible_name]:
                start = annotation[start_name]
                end = annotation[end_name]

                overlap = self._calculate_interval_overlap(clip, (start, end))

                if self._overlap_over_threshold(
                    overlap, self.threshold, clip[1] - clip[0], self.absolute_threshold
                ):
                    labeled_time += overlap
                    labels.append(idx)

        # If we didn't account for the full clip length with the labeled actions
        # we add the `other` class
        if labeled_time < clip[1] - clip[0]:
            labels.append(self.label_description.other_class)
        return labels

    def _calculate_interval_overlap(
        self, interval1: IntervalInSeconds, interval2: IntervalInSeconds
    ) -> float:
        if (
            interval1[0] <= interval2[1] and interval1[1] >= interval2[0]
        ):  # If the intervals overlap
            return max(
                0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
            )
        else:  # If the intervals do not overlap
            return 0.0

    def _overlap_over_threshold(
        self, overlap: float, threshold: float, clip_length: float, absolute: bool
    ) -> bool:
        if absolute:
            return overlap >= threshold
        else:
            return overlap / clip_length >= threshold
