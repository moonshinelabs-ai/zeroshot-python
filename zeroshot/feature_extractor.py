import os

import numpy as np

from .downloader import fetch_model

_MODELS_ONNX = {
    "dinov2_small": "https://zeroshot-prod-models.s3.us-west-2.amazonaws.com/dinov2_onnx/dinov2_small.onnx",
    "dinov2_base": "",
}

_MODEL_TO_TORCH = {
    "dinov2_small": "dinov2_vits14",
}

_SUPPORTED_BACKENDS = ["onnx", "torch"]


class FeatureExtractor(object):
    def _get_onnx_model(self, name: str) -> str:
        import onnxruntime

        # If the model is a file path, just use that model. Otherwise download it.
        if os.path.exists(name):
            self.path = name
        else:
            # Check that the model is valid.
            if name not in _MODELS_ONNX:
                raise ValueError(
                    f"Model {name} not found. Possible values are {_MODELS_ONNX.keys()}"
                )
            self.path = fetch_model(_MODELS_ONNX[name])

        # Load the model via ONNX.
        try:
            self.model = onnxruntime.InferenceSession(
                self.path, providers=["CUDAExecutionProvider"]
            )
        except:
            self.model = onnxruntime.InferenceSession(self.path)

    def _get_torch_model(self, name: str) -> str:
        import torch

        newname = _MODEL_TO_TORCH[name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("facebookresearch/dinov2", newname).to(self.device)

    def _run_onnx_model(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError(
                "Image must be 3-dimensional, batch not supported with ONNX."
            )

        # Rotate the dimensions so that channels is first.
        image = np.transpose(image, (2, 0, 1))

        # Make sure the image is a float32 and has a batch dimension.
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        # Process the model and keep the batch dimension.
        outputs = self.model.run(None, {"input": image})
        return outputs[0]

    def _run_torch_model(self, image: np.ndarray) -> np.ndarray:
        import torch

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Make sure the channels is first
        image = np.transpose(image, (0, 3, 1, 2))

        # Convert to a torch tensor.
        tensor = torch.from_numpy(image).float().to(self.device)

        # Run the model
        with torch.no_grad():
            outputs = self.model(tensor).cpu().numpy()
        return outputs

    def __init__(self, name: str, backend="onnx"):
        """Create a feature extractor.

        Args:
            name (str): The name of the model to use, or a path directly to the model.
        """
        self.name = name
        self.path = None
        self.device = "cpu"
        self.backend = backend

        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Backend must be {', '.join(_SUPPORTED_BACKENDS)}")

        if self.backend == "onnx":
            self._get_onnx_model(name)
        elif self.backend == "torch":
            self._get_torch_model(name)
        else:
            raise ValueError(f"Backend {self.backend} not supported.")

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process an image into a feature vector.

        Args:
            image (np.ndarray): The image to process, in RGB format with channels last.

        Returns:
            np.ndarray: The feature vector.
        """
        if self.backend == "onnx":
            return self._run_onnx_model(image)
        elif self.backend == "torch":
            return self._run_torch_model(image)
        else:
            raise ValueError(f"Backend {self.backend} not supported.")


class DINOV2FeatureExtractor(FeatureExtractor):
    def __init__(self, size: str = "small", backend: str = "onnx"):
        """Create a DINO v2 feature extractor.

        Args:
            size (str): The size of the model to use, either 'small' or 'base'.
        """
        name = f"dinov2_{size}"
        super().__init__(name, backend=backend)
