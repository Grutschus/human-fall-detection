import pandas as pd
from datasets.high_quality_fall_dataset import HighQualityFallDataset
from datasets.transforms.sampling_strategy import UniformSampling
from datasets.transforms.label_strategy import ExistenceLabel
import unittest
from unittest import mock


class TestHighQualityFallDataset(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_load_data_list(self):
        # Create mock objects for the sampling strategy and label strategy
        ss = mock.create_autospec(UniformSampling)
        ls = mock.create_autospec(ExistenceLabel)

        # Patch the necessary methods of the sampling strategy and label strategy
        ss.sample.return_value = [(0, 1), (1, 2)]
        ls.label.return_value = 1

        mock_dataframe = pd.DataFrame(
            {
                "video_path": ["video1.avi", "video2.avi"],
                "fall_start": [1, 2],
                "fall_end": [3, 4],
                "lying_start": [3, 4],
                "lying_end": [7, 8],
                "length": [9, 10],
                "fall_visible": [True, False],
                "lying_visible": [True, False],
            }
        )

        with mock.patch.object(
            pd,
            "read_csv",
            return_value=mock_dataframe,
        ):
            hqfd = HighQualityFallDataset(
                "tests/test_data/test_annotation.csv", ss, ls, []
            )
            data_list = hqfd.load_data_list()

        assert data_list == [
            {"filename": "video1.avi", "label": 1, "interval": (0, 1)},
            {"filename": "video1.avi", "label": 1, "interval": (1, 2)},
            {"filename": "video2.avi", "label": 1, "interval": (0, 1)},
            {"filename": "video2.avi", "label": 1, "interval": (1, 2)},
        ]


if __name__ == "__main__":
    unittest.main()
