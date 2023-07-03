import argparse
import os
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image
from sklearn.linear_model import LogisticRegression  # type: ignore

from zeroshot import Classifier, create_preprocess_fn


def url_to_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def golden_test(target: str, classifier: Any) -> np.ndarray:
    sklearn_model = LogisticRegression()
    sklearn_model.coef_ = classifier.linear_model.coefs
    sklearn_model.intercept_ = classifier.linear_model.intercept
    sklearn_model.classes_ = np.arange(len(classifier.classes))

    # Load the image into a numpy array.
    if target.startswith("http"):
        image = url_to_image(target)
    else:
        image = Image.open(target)
    image_np = np.array(image)[:, :, 0:3]

    # Preprocess the image using the classifier.
    image_np = classifier.preprocess_fn(image_np)

    # Extract the features from the image using the classifier.
    features = classifier.feature_extractor.process(image_np)

    # Run the classifier.
    return sklearn_model.predict_proba(features)


def run_test(target: str, classifier: Any) -> None:
    # Run the classifier.
    prediction = classifier.predict(target)
    prediction_probs = classifier.predict_proba(target)

    # Print the results.
    prediction_str = classifier.classes[prediction]
    print(f"Predicted class: {prediction_str}")

    probabilities_str = zip(classifier.classes, prediction_probs)
    print(f"Predicted probabilities:")
    for class_name, probability in probabilities_str:
        if prediction_str == class_name:
            print("\033[1m", end="")
        print(f"\t{class_name}: {probability:.2%}")
        if prediction_str == class_name:
            print("\033[0m", end="")

    # Check the results using sklearn.
    expected_probs = golden_test(target, classifier).ravel()
    assert prediction_str == "a photo of a zoo animal"
    for expected, prob in zip(expected_probs, prediction_probs):
        assert np.isclose(expected, prob, atol=1e-5)


def run_test_binary(target: str, classifier: Any) -> None:
    # Run the classifier.
    prediction = classifier.predict(target)
    prediction_probs = classifier.predict_proba(target)

    # Run the sklearn version
    comparison = golden_test(target, classifier).ravel()

    # Check the results.
    for expected, prob in zip(comparison, prediction_probs):
        assert np.isclose(expected, prob, atol=1e-5)


def main(args: argparse.Namespace) -> None:
    classifier = Classifier(os.path.join(args.test_file_path, "test_model.json"))
    run_test(os.path.join(args.test_file_path, "giraffe.png"), classifier)
    run_test(
        "https://moonshine-assets.s3.us-west-2.amazonaws.com/giraffe.png", classifier
    )

    classifier = Classifier(os.path.join(args.test_file_path, "test_binary.json"))
    run_test_binary(os.path.join(args.test_file_path, "giraffe.png"), classifier)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--test_file_path", default="scripts/test_files", type=str)
    known_args = args.parse_args()

    main(known_args)
