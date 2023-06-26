import io
from typing import Any, Optional, Union

import numpy as np
import requests
from PIL import Image


def _from_data_or_path(input: Union[str, io.BytesIO]) -> np.ndarray:
    """Loads the model from either a path or data."""
    img = Image.open(input)
    img = img.convert("RGB")
    img_data = np.array(img)

    return img_data


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
