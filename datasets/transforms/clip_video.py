from typing import Dict
from mmaction.registry import TRANSFORMS  # type: ignore
from mmcv.transforms import BaseTransform  # type: ignore


@TRANSFORMS.register_module()
class ClipVideo(BaseTransform):
    """Clip a video to a given interval.
    Does not affect the video_reader. Just sets the `total_frames` and
    `start_index` keys.

    If a `start_index` key is already present, it will be offset.

    Required Keys:

        - `interval`: a tuple of two floats,
            the start and end of the interval in seconds.
        - `avg_fps`

    Modified Keys:

        - `start_index`
        - `total_frames`
    """

    def transform(self, results: Dict) -> Dict:
        """Perform the `ClipVideo` transformation.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        interval = results["interval"]
        total_frames = results["total_frames"]
        fps = results["avg_fps"]
        offset = results["start_index"] if "start_index" in results else 0
        start_frame = int(interval[0] * fps) + offset
        end_frame = min(int(interval[1] * fps) + offset, total_frames)
        results["start_index"] = start_frame
        results["total_frames"] = end_frame - start_frame
        return results
