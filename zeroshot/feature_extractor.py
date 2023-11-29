import os
import warnings

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
    def _get_onnx_model(self, name: str):
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
            with warnings.catch_warnings():
                # Ignore the CUDA warning...
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = onnxruntime.InferenceSession(
                    self.path, providers=["CUDAExecutionProvider"]
                )
        except:
            self.model = onnxruntime.InferenceSession(self.path)

    def _get_torch_model(self, name: str):
        import torch

        newname = _MODEL_TO_TORCH[name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():
            # Ignore the CUDA warning from xformers
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model = torch.hub.load("facebookresearch/dinov2", newname).to(
                self.device
            )

    def _run_onnx_model(
        self, image: np.ndarray, feature_map: bool = False
    ) -> np.ndarray:
        if feature_map:
            raise NotImplementedError("Feature map not supported with ONNX.")

        # Process the model and keep the batch dimension.
        outputs = self.model.run(None, {"input": image})
        return outputs[0]

    def _run_torch_model(self, image: np.ndarray, feature_map=False) -> np.ndarray:
        import torch

        # Get patch sizes.
        _, _, h, w = image.shape
        patches_h, patches_w = h // 14, w // 14

        # Convert to a torch tensor.
        tensor = torch.from_numpy(image).float().to(self.device)

        # Run the model
        with torch.no_grad():
            if feature_map:
                intermediates = self.model.get_intermediate_layers(tensor, n=1)[0]
                np_intermediates = intermediates.cpu().numpy()
                return np.reshape(np_intermediates, (1, patches_h, patches_w, -1))
            else:
                return self.model(tensor).cpu().numpy()

    def __init__(self, name: str, backend="onnx"):
        """Create a feature extractor.

        Args:
            name (str): The name of the model to use, or a path directly to the model.
        """
        self.name = name
        self.path = ""
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

    def process(self, image: np.ndarray, feature_map: bool = False) -> np.ndarray:
        """Process an image into a feature vector.

        Args:
            image (np.ndarray): The image to process, in RGB format with channels last.

        Returns:
            np.ndarray: The feature vector.
        """
        # If the image is 3-dimensional, add a batch dimension.
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Make sure the channels is first
        image = np.transpose(image, (0, 3, 1, 2))

        # For now we only support batch size 1, since ONNX doesn't allow variable sizes.
        b, c, h, w = image.shape
        if b != 1:
            raise ValueError("Batch size must be 1 in feature extractor.")

        # Check that the patch sizes are 14
        if h % 14 != 0 or w % 14 != 0:
            raise ValueError(
                "Image size must be divisible by patch size (14) in feature extractor."
            )

        # Ensure we're float32
        image = image.astype(np.float32)

        if self.backend == "onnx":
            return self._run_onnx_model(image, feature_map=feature_map)
        elif self.backend == "torch":
            return self._run_torch_model(image, feature_map=feature_map)
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
