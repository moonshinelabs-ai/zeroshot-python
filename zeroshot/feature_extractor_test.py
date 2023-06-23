import os
import unittest
from unittest.mock import patch

from .feature_extractor import DINOV2FeatureExtractor, FeatureExtractor


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


if __name__ == "__main__":
    unittest.main()
