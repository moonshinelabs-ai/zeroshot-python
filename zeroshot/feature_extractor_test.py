import os
import unittest
from unittest.mock import patch

import numpy as np
from parameterized import parameterized

from .feature_extractor import DINOV2FeatureExtractor, FeatureExtractor

# Force CPU only for this test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XFORMERS_DISABLED"] = "1"


class TestFeatureExtractor(unittest.TestCase):
    @patch("os.path.exists")
    def test_invalid_model_name(self, mock_exists):
        """Test if the model name isn't valid."""
        mock_exists.return_value = False

        with self.assertRaises(ValueError):
            FeatureExtractor("invalid_name")

    @patch("os.path.exists")
    def test_invalid_file_path(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(ValueError):
            FeatureExtractor("/invalid/path/to/model")

    @parameterized.expand(
        [
            ((140, 140),),
            ((224, 224),),
            ((448, 448),),
            ((224, 448),),
        ]
    )
    def test_small_dino_single_batch_torch_gives_tensor(
        self, test_shape: tuple[int, int]
    ):
        feature_extractor = DINOV2FeatureExtractor("small", backend="torch")

        # Create a random image that's already normalized
        image = np.random.rand(1, test_shape[0], test_shape[1], 3)
        features = feature_extractor.process(image)

        self.assertSequenceEqual(features.shape, (1, 384))

    def test_small_dino_single_batch_onnx_gives_tensor(self):
        feature_extractor = DINOV2FeatureExtractor("small")

        # Create a random image that's already normalized
        image = np.random.rand(1, 224, 224, 3)
        features = feature_extractor.process(image)

        self.assertSequenceEqual(features.shape, (1, 384))

    def test_small_dino_wrong_shape_onnx_raises(self):
        feature_extractor = DINOV2FeatureExtractor("small")

        image = np.random.rand(1, 448, 448, 3)
        with self.assertRaises(Exception):
            features = feature_extractor.process(image)

    def test_small_dino_wrong_batch_raises(self):
        feature_extractor = DINOV2FeatureExtractor("small")

        image = np.random.rand(4, 224, 224, 3)
        with self.assertRaises(ValueError):
            features = feature_extractor.process(image)


if __name__ == "__main__":
    unittest.main()
