import os

import numpy as np
import onnxruntime  # type: ignore

from .downloader import fetch_model

MODELS = {
    "dinov2_small": "https://zeroshot-prod-models.s3.us-west-2.amazonaws.com/dinov2_onnx/dinov2_small.onnx",
    "dinov2_base": "",
}


class FeatureExtractor(object):
    def __init__(self, name: str):
        """Create a feature extractor.

        Args:
            name (str): The name of the model to use, or a path directly to the model.
        """
        self.name = name
        self.path = None

        # If the model is a file path, just use that model. Otherwise download it.
        if os.path.exists(name):
            self.path = name
        else:
            # Check that the model is valid.
            if name not in MODELS:
                raise ValueError(
                    f"Model {name} not found. Possible values are {MODELS.keys()}"
                )
            self.path = fetch_model(MODELS[name])

        # Load the model via ONNX.
        self.model = onnxruntime.InferenceSession(self.path)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process an image into a feature vector.

        Args:
            image (np.ndarray): The image to process, in RGB format with channels last.

        Returns:
            np.ndarray: The feature vector.
        """
        # Rotate the dimensions so that channels is first.
        image = np.transpose(image, (2, 0, 1))

        # Make sure the image is a float32 and has a batch dimension.
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        # Process the model and keep the batch dimension.
        outputs = self.model.run(None, {"input": image})
        return outputs[0]


class DINOV2FeatureExtractor(FeatureExtractor):
    def __init__(self, size: str = "small"):
        """Create a DINO v2 feature extractor.

        Args:
            size (str): The size of the model to use, either 'small' or 'base'.
        """
        name = f"dinov2_{size}"
        super().__init__(name)
