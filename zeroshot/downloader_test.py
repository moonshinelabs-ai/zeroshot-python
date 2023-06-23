import os
import tempfile

import pytest
import requests_mock

from .downloader import fetch_model


def test_fetch_model_downloads_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        url = "http://usezeroshot.com/model"
        expected_content = b"model file"

        with requests_mock.Mocker() as m:
            # Request the file via mocker.
            m.get(url, content=expected_content)

            # Download and check that it worked.
            file_path = fetch_model(url, tmpdir)
            with open(file_path, "rb") as f:
                assert f.read() == expected_content


def test_fetch_model_uses_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        url = "http://usezeroshot.com/model"
        expected_content = b"model file"

        with requests_mock.Mocker() as m:
            # Request the file via mocker.
            m.get(url, content=expected_content)

            # We'll download twice to check the cache.
            file_path = fetch_model(url, tmpdir)
            file_path_2 = fetch_model(url, tmpdir)

            # Check that the file was not re-downloaded.
            assert m.call_count == 1

            # The file should be the same each time.
            assert file_path == file_path_2
