# Zeroshot (Python)
Image classification for the masses

## Installation
Install via pip: `pip install zeroshot`

## Usage
First, go to usezeroshot.com and create a classifier. See [here]() for more instructions.

Then, in Python:

```python
from zeroshot import Classifier, create_preprocess_fn

image = ... # This should be an RGB numpy image with channels last.

# Create the classifier and preprocessing function.
classifier = Classifier("your model string or path")
preprocess_fn = create_preprocess_fn("dino")

# Run the model!
prediction = classifier.predict(preprocess_fn(image))
print(f"The image is class {prediction}")
```

## Read the docs
PUT DOCS HERE.