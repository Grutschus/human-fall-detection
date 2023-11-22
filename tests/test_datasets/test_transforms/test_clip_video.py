import unittest
from mmaction.datasets.transforms.loading import DecordDecode, DecordInit, SampleFrames
from mmaction.registry import DATASETS
from datasets import HighQualityFallDataset
from datasets.transforms.clip_video import ClipVideo
from datasets.transforms.label_strategy import ExistenceLabel, HQFD_LABEL_DESCRIPTION
from datasets.transforms.sampling_strategy import UniformSampling


class TestClipVideo(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_clip_video(self):
        sampling_strategy = UniformSampling(clip_len=10)
        label_strategy = ExistenceLabel(HQFD_LABEL_DESCRIPTION)

        dataset_cfg = dict(
            type=HighQualityFallDataset,
            sampling_strategy=sampling_strategy,
            label_strategy=label_strategy,
            ann_file="tests/test_data/test_annotation.csv",
            pipeline=[
                dict(type=DecordInit),
                dict(type=ClipVideo),
                dict(
                    type=SampleFrames,
                    clip_len=16,
                    frame_interval=4,
                    num_clips=5,
                    test_mode=True,
                ),
                dict(type=DecordDecode),
            ],
            multi_class=True,
            num_classes=3,
            test_mode=True,
        )

        dataset = DATASETS.build(dataset_cfg)

        first_sample = dataset[0]

        # Just some examples. The main purpose here was
        # to check that the pipeline was working correctly
        assert len(first_sample["imgs"]) == 16 * 5  # 16 frames per clip, 5 clips
        assert (
            max(first_sample["frame_inds"]) <= 10 * first_sample["avg_fps"]
        )  # 10 seconds of video


if __name__ == "__main__":
    unittest.main()
