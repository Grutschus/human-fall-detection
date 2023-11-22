import unittest
import pandas as pd

from datasets.transforms.sampling_strategy import UniformSampling


class TestUniformSampling(unittest.TestCase):
    def setUp(self):
        self.annotation = pd.Series({"length": 10})

    def test_no_overlap(self):
        sampler = UniformSampling(clip_len=2.0, stride=0.0, overlap=False)
        expected_samples = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
        self.assertEqual(sampler.sample(self.annotation), expected_samples)

    def test_with_overlap(self):
        sampler = UniformSampling(clip_len=2.0, stride=1.0, overlap=True)
        expected_samples = [
            (0.0, 2.0),
            (1.0, 3.0),
            (2.0, 4.0),
            (3.0, 5.0),
            (4.0, 6.0),
            (5.0, 7.0),
            (6.0, 8.0),
            (7.0, 9.0),
            (8.0, 10.0),
        ]
        self.assertEqual(sampler.sample(self.annotation), expected_samples)

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            sampler = UniformSampling(clip_len=2.0, stride=0.0, overlap=True)
            sampler.sample(self.annotation)


if __name__ == "__main__":
    unittest.main()
