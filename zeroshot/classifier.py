import json
import re
import urllib.request
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from .feature_extractor import DINOV2FeatureExtractor
from .logistic_regression import LogisticRegression

API_ENDPOINT = "https://api.wanpan.rest"
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def _load_from_guid(guid: str) -> dict[str, Any]:
    """Loads the model from a guid."""
    # Fetch the model from the API
    fetch_endpoint = f"{API_ENDPOINT}/get_classifier/{guid}"
    with urllib.request.urlopen(fetch_endpoint) as response:
        data = json.load(response)
    return data


def _load_from_file(path: str) -> dict[str, Any]:
    """Loads the model from a file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def _infer_path_type(path: str) -> str:
    """Infers the path type from the path."""
    uuid_pattern = re.compile(UUID_PATTERN, re.IGNORECASE)
    if uuid_pattern.match(path):
        return "guid"
    else:
        return "file"


def _numpy_from_path(path: str) -> np.ndarray:
    """Get a numpy array from a path."""
    img = Image.open(path)
    img = img.convert("RGB")
    img_data = np.array(img)

    return img_data


class Classifier(object):
    def _load_from_data(self, data: dict) -> None:
        # Load the model
        self.linear_model = LogisticRegression(
            coefs=np.array(data["coefficients"]), intercept=np.array(data["intercepts"])
        )
        self.class_list = data["class_list"]
        self.feature_extractor_name = data["feature_extractor"]

        self.feature_extractor = DINOV2FeatureExtractor(self.feature_extractor_name)

    def __init__(self, path: str, path_type: str = "infer"):
        # Check that the path type is valid.
        possible_types = ("infer", "guid", "file")
        if path_type not in possible_types:
            raise ValueError(
                f"Path type must be one of {possible_types}, got {path_type}"
            )

        self.path = path
        self.class_list = []

        # By default we'll just infer the type of path. If it matches a UUID
        # then we'll assume it's a GUID. Since in theory there could be a GUID
        # in the file path, we'll allow the user to override this inference.
        if path_type == "infer":
            path_type = _infer_path_type(path)

        if path_type == "file":
            data = _load_from_file(self.path)
        elif path_type == "guid":
            data = _load_from_guid(self.path)
        self._load_from_data(data)

    def predict(self, image: Union[str, np.ndarray]) -> int:
        """Predicts the class of an image.

        Args:
            image (Union[str, np.ndarray]): The image to predict, either a url or a numpy array.

        Returns:
            int: The predicted class
        """
        if isinstance(image, str):
            image_np = _numpy_from_path(image)
        elif isinstance(image, np.ndarray):
            image_np = image

        features = self.feature_extractor.process(image_np)
        return self.linear_model.predict(features)[0]

    def predict_proba(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Predicts the probabilities of all classes.

        Args:
            image (Union[str, np.ndarray]): The image to predict, either a url or a numpy array.

        Returns:
            np.ndarray: The predicted class probs.
        """
        if isinstance(image, str):
            image_np = _numpy_from_path(image)
        elif isinstance(image, np.ndarray):
            image_np = image

        features = self.feature_extractor.process(image_np)
        return self.linear_model.predict_proba(features)[0]
