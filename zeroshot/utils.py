import contextlib
import io
import sys

import numpy as np
import requests
from PIL import Image


def _from_data_or_path(input: str | io.BytesIO) -> np.ndarray:
    """Loads the model from either a path or data."""
    img = Image.open(input)
    img = img.convert("RGB")
    img_data = np.array(img)

    return img_data


class nostderr:
    def __init__(self):
        self._original_stderr = None

    def __enter__(self):
        import os

        # Save the original stderr so it can be restored later
        self._original_stderr = sys.stderr

        # Redirect stderr to null device to suppress output
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the redirected stderr and restore the original one
        sys.stderr.close()
        sys.stderr = self._original_stderr


def numpy_from_path(path: str) -> np.ndarray:
    """Get a numpy array from a path."""
    return _from_data_or_path(path)


def numpy_from_url(path: str) -> np.ndarray:
    """Get a numpy array from a path."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0;Win64) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }
    response = requests.get(path, headers=headers, timeout=1)
    data = io.BytesIO(response.content)
    return _from_data_or_path(data)
