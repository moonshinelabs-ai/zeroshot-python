import argparse
from typing import Any

import numpy as np
from PIL import Image

from zeroshot import Classifier, create_preprocess_fn


def run_test(target: str, classifier: Any) -> None:
    # Run the classifier.
    prediction = classifier.predict(target)
    prediction_probs = classifier.predict_proba(target)

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
    assert prediction_str == "a photo of a zoo animal"
    for expected, prob in zip(
        [0.05371324, 0.03506953, 0.04463777, 0.86657947], prediction_probs
    ):
        assert np.isclose(expected, prob, atol=1e-5)


def main(args: argparse.Namespace) -> None:
    classifier = Classifier(args.model)
    run_test(args.image, classifier)
    run_test(
        "https://moonshine-assets.s3.us-west-2.amazonaws.com/giraffe.png", classifier
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", default="scripts/test_files/test_model.json", type=str)
    args.add_argument("--image", default="scripts/test_files/giraffe.png", type=str)
    known_args = args.parse_args()

    main(known_args)
