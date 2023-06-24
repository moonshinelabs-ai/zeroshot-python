import argparse

import numpy as np
from PIL import Image

from zeroshot import Classifier, create_preprocess_fn


def load_image(path: str) -> np.ndarray:
    # Load and convert to Numpy array.
    img_pil = Image.open(path)
    img = np.array(img_pil)

    # Remove the alpha channel if there is one.
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # BGR to RGB
    return img[..., ::-1]


def main(args: argparse.Namespace) -> None:
    classifier = Classifier(args.model)
    preprocess_fn = create_preprocess_fn("dino")

    # Load an image into a numpy array.
    image = load_image(args.image)
    preprocessed_image = preprocess_fn(image)

    # Run the classifier.
    prediction = classifier.predict(preprocessed_image)
    prediction_probs = classifier.predict_proba(preprocessed_image)

    # Print the results.
    prediction_str = classifier.class_list[prediction]
    print(f"Predicted class: {prediction_str}")

    probabilities_str = zip(classifier.class_list, prediction_probs)
    print(f"Predicted probabilities:")
    for class_name, probability in probabilities_str:
        if prediction_str == class_name:
            print("\033[1m", end="")
        print(f"\t{class_name}: {probability:.2%}")
        if prediction_str == class_name:
            print("\033[0m", end="")

    # Check the results.
    assert prediction_str == "a photo of a bus"
    for expected, prob in zip(
        [0.00330515, 0.002696, 0.98795263, 0.00604621], prediction_probs
    ):
        assert np.isclose(expected, prob, atol=1e-5)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", default="scripts/test_files/test_model.json", type=str)
    args.add_argument("--image", default="scripts/test_files/bus.png", type=str)
    known_args = args.parse_args()

    main(known_args)
