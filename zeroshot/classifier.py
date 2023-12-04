import json
import re
import urllib.request
from typing import Any

import numpy as np

from .feature_extractor import DINOV2FeatureExtractor
from .logistic_regression import LogisticRegression
from .preprocessing import create_preprocess_fn
from .utils import numpy_from_path, numpy_from_url

API_ENDPOINT = "https://dvnnfiycsg.execute-api.us-west-2.amazonaws.com/staging"
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def _load_from_guid(guid: str) -> dict[str, Any]:
    """Loads the model from a guid."""
    # Fetch the model from the API
    fetch_endpoint = f"{API_ENDPOINT}/classifiers/{guid}"
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


class Classifier(object):
    def _load_from_data(self, data: dict) -> None:
        # Load the model
        self.linear_model = LogisticRegression(
            coefs=np.array(data["coefficients"]), intercept=np.array(data["intercepts"])
        )
        self.classes = data["class_list"]
        self.feature_extractor_name = data["feature_extractor"]

        self.feature_extractor = DINOV2FeatureExtractor(
            self.feature_extractor_name, backend=self.backend
        )

    def _image_from_str(self, image: str | np.ndarray) -> np.ndarray:
        """Generate feature vector from a string."""
        if isinstance(image, str) and image.startswith("http"):
            image_np = numpy_from_url(image)
        elif isinstance(image, str):
            image_np = numpy_from_path(image)
        elif isinstance(image, np.ndarray):
            image_np = image

        # Preprocess the image if necessary.
        if self.preprocess_fn is not None:
            image_np = self.preprocess_fn(image_np)

        return image_np

    def __init__(
        self,
        path: str,
        path_type: str = "infer",
        preprocessor: str | None = "dino",
        backend: str = "onnx",
    ):
        # Check that the path type is valid.
        possible_types = ("infer", "guid", "file")
        if path_type not in possible_types:
            raise ValueError(
                f"Path type must be one of {possible_types}, got {path_type}"
            )

        self.path = path
        self.classes = []
        self.backend = backend

        # Load the preprocessor for the model.
        if preprocessor is None:
            self.preprocess_fn = None
        else:
            self.preprocess_fn = create_preprocess_fn(preprocessor)

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

    def predict(self, image: str | np.ndarray) -> int:
        """Predicts the class of an image.

        Args:
            image: The image to predict, either a url or a numpy array.

        Returns:
            The predicted class
        """
        image = self._image_from_str(image)
        features = self.feature_extractor.process(image)
        return self.linear_model.predict(features)[0]

    def predict_patches(self, image: str | np.ndarray) -> np.ndarray:
        """Predicts the class of an image for each patch

        Args:
            image: The image to predict, either a url or a numpy array.

        Returns:
            The predicted class
        """
        image = self._image_from_str(image)
        features = self.feature_extractor.process(image, feature_map=True)

        # Predict a class for each patch.
        predictions = np.zeros((features.shape[1], features.shape[2]))
        for i in range(features.shape[1]):
            for j in range(features.shape[2]):
                patch = features[:, i, j, :]
                predictions[i, j] = self.linear_model.predict(patch)

        return predictions

    def predict_proba(self, image: str | np.ndarray) -> np.ndarray:
        """Predicts the probabilities of all classes.

        Args:
            image: The image to predict, either a url or a numpy array.

        Returns:
            The predicted class probs.
        """
        image = self._image_from_str(image)
        features = self.feature_extractor.process(image)
        return self.linear_model.predict_proba(features)[0]
