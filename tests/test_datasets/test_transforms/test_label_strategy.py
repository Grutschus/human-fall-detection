import pandas as pd
from datasets.transforms.label_strategy import ExistenceLabel, LabelDescription
from datasets.transforms.label_strategy import (
    _calculate_interval_overlap,
    _overlap_over_threshold,
)
import unittest

mock_label_description = LabelDescription(
    names=["fall", "other"],
    start_timestamp_names=["fall_start"],
    end_timestamp_names=["fall_end"],
    visible_names=["fall_visible"],
    other_class=1,
)


class TestExistanceLabel(unittest.TestCase):
    def test_calculate_interval_overlap(self):
        assert _calculate_interval_overlap((1, 5), (4, 6)) == 1
        assert _calculate_interval_overlap((1, 5), (6, 10)) == 0

    def test_overlap_over_threshold(self):
        assert _overlap_over_threshold(5, 3, 10, True)
        assert not _overlap_over_threshold(5, 6, 10, True)
        assert _overlap_over_threshold(5, 0.4, 10, False)
        assert not _overlap_over_threshold(5, 0.6, 10, False)
        assert not _overlap_over_threshold(0, 0, 10, True)
        assert not _overlap_over_threshold(0, 0, 10, False)

    def test_label(self):
        el = ExistenceLabel(mock_label_description)
        annotation_visible = pd.Series(
            {
                "fall_start": 3,
                "fall_end": 5,
                "fall_visible": True,
                "other_start": 6,
                "other_end": 10,
                "other_visible": False,
            }
        )

        annotation_invisible = pd.Series(
            {
                "fall_start": 3,
                "fall_end": 5,
                "fall_visible": False,
            }
        )

        assert el.label(annotation_visible, (3, 5)) == [0]
        assert el.label(annotation_visible, (2, 6)) == [0, 1]
        assert el.label(annotation_invisible, (3, 5)) == [1]


if __name__ == "__main__":
    unittest.main()
