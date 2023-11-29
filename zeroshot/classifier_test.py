import json
import re
import unittest
import urllib.request
from unittest.mock import mock_open, patch

from .classifier import _infer_path_type, _load_from_file, _load_from_guid


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


if __name__ == "__main__":
    unittest.main()
