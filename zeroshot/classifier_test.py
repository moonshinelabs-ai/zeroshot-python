import json
import os
import re
import unittest
import urllib.request
from unittest.mock import mock_open, patch

import numpy as np

from .classifier import (Classifier, _infer_path_type, _load_from_file,
                         _load_from_guid)

# Force CPU only for this test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XFORMERS_DISABLED"] = "1"


class TestClassifier(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_load_from_guid(self, mock_urlopen):
        # Mock API response
        mock_api_response = {"model": "model data"}
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            mock_api_response
        ).encode()

        guid = "12345678-1234-5678-1234-567812345678"
        expected_result = mock_api_response

        result = _load_from_guid(guid)

        self.assertEqual(result, expected_result)

    @patch("builtins.open", new_callable=mock_open, read_data='{"model": "model data"}')
    def test_load_from_file(self, mock_open):
        # Mock file path
        mock_file_path = "/path/to/model.json"
        with open(mock_file_path) as f:
            expected_result = json.load(f)

        result = _load_from_file(mock_file_path)

        self.assertEqual(result, expected_result)

    def test_infer_path_type(self):
        guid = "12345678-1234-5678-1234-567812345678"
        file_path = "/path/to/model.json"

        self.assertEqual(_infer_path_type(guid), "guid")
        self.assertEqual(_infer_path_type(file_path), "file")

    def test_load_classifier_from_file_onnx_backend(self):
        model_path = "test_files/test_model.json"
        # Get the full path
        full_path = os.path.join(os.path.dirname(__file__), model_path)

        classifier = Classifier(full_path, backend="onnx")

        # Feed a random image to the classifier (0-255)
        np.random.seed(0)
        rand_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        result = classifier.predict(rand_image)

        self.assertEqual(result, 1)

    def test_load_classifier_from_file_torch_backend(self):
        model_path = "test_files/test_model.json"
        # Get the full path
        full_path = os.path.join(os.path.dirname(__file__), model_path)

        classifier = Classifier(full_path, backend="torch")

        # Feed a random image to the classifier (0-255)
        np.random.seed(0)
        rand_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        result = classifier.predict(rand_image)

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
