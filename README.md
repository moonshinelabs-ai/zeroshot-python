# Zeroshot (Python)

Image classification for the masses

## Installation

Install via pip: `pip install zeroshot`

## Usage

First, go to usezeroshot.com and create a classifier. See [here](<>) for more instructions.

Then, in Python (`image` should be an RGB numpy array with channels last):

```python
import zeroshot

# Create the classifier and preprocessing function.
classifier = zeroshot.Classifier("your model string or path")
preprocess_fn = zeroshot.create_preprocess_fn()

# Run the model!
prediction = classifier.predict(preprocess_fn(image))
print(f"The image is class {prediction}")
```

## Read the docs

See the [docs](docs/getting_started.md) folder for some details.
