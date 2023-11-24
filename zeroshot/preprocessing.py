from typing import Callable

import numpy as np
from PIL import Image


def resize(image, new_height, new_width):
    """Resize an image using PIL."""
    img = Image.fromarray(image)
    img = img.resize((new_width, new_height), resample=Image.BICUBIC)
    resized_image = np.array(img)

    return resized_image


def center_crop(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    height, width, _ = image.shape

    # Compute the start coords and crop.
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped_image = image[start_y : start_y + new_height, start_x : start_x + new_width]

    return cropped_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # Ensure the image has three channels (RGB)
    if image.shape[-1] != 3:
        raise ValueError("Input image must have three channels (RGB)")

    # Normalize the image
    normalized_image = (image - imagenet_mean) / imagenet_std

    return normalized_image


def resize_image_fixed_side(
    image: np.ndarray, side_len: int, method: str = "max"
) -> np.ndarray:
    # Need epsilon for floating point errors.
    epsilon = 1e-3
    height, width, _ = image.shape

    # Compute the scaling factor.
    if method == "max":
        scaling_factor = side_len / max(height, width)
    elif method == "min":
        scaling_factor = side_len / min(height, width)
    else:
        raise ValueError("Invalid method specified. Choose 'max' or 'min'.")

    # Compute the new dimensions plus an epsilon that will get truncated.
    new_height = int(height * scaling_factor + epsilon)
    new_width = int(width * scaling_factor + epsilon)

    assert new_height == side_len or new_width == side_len, "One side incorrect"

    resized_image = resize(image, new_height, new_width)

    return resized_image


def resize_image_min_side(image: np.ndarray, min_side_len: int = 224) -> np.ndarray:
    return resize_image_fixed_side(image, min_side_len, method="min")


def resize_image_max_side(image: np.ndarray, max_side_len: int = 224) -> np.ndarray:
    return resize_image_fixed_side(image, max_side_len, method="max")


def _dino_preprocess(image: np.ndarray) -> np.ndarray:
    """Do standardization and resizing for this model.

    We will do ImageNet standardization, and resize the shortest size to 224, then crop the center 224x224.

    Args:
        image (np.ndarray): The image to preprocess.

    Returns:
        np.ndarray: The preprocessed image.
    """
    resized_image = resize_image_min_side(image, 224)
    cropped_image = center_crop(resized_image, 224, 224)

    return normalize_image(cropped_image)


def create_preprocess_fn(name: str = "dino") -> Callable[[np.ndarray], np.ndarray]:
    """Creates a preprocessing function from a name."""
    if name == "dino":
        return _dino_preprocess
    else:
        raise ValueError(f"Unknown preprocessing function {name}")
