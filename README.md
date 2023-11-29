# Zeroshot (Python)

Image classification for the masses

## Installation

Install via pip: `pip install zeroshot`

For GPU support, `pip install zeroshot[torch]`

N.B. In theory ONNX supports GPU, but the restrictions on CUDA version are iffy at best, and so for easiest results just use PyTorch. If you're brave, instead `pip install onnxruntime-gpu`.

## Usage

First, go to app.usezeroshot.com and create a classifier. Check out the video on the [landing page](usezeroshot.com) for an example.

Then, in Python (`image` should be an RGB numpy array with channels last):

```python
import zeroshot

# Create the classifier and preprocessing function.
classifier = zeroshot.Classifier("model-uuid-goes-here")
preprocess_fn = zeroshot.create_preprocess_fn()

# Run the model!
prediction = classifier.predict(preprocess_fn(image))
print(f"The image is class {prediction}")
```

You can also download the classifier and save it somewhere locally so you don't need to hit the server each time. Hit "download model" in the web-app and save the json file somewhere. You can then instead do:

```python
classifier = zeroshot.Classifier("/home/user/path/to/model.json")
```

## Additional Tips

* To use a GPU, install the torch backend with `pip install zeroshot[torch]`
* If you are hitting issues with torch trying to run on CPU, try disabling XFormers by setting XFORMERS_DISABLED=1 in your ENV varaibles.

## Read the docs

See the [docs](https://github.com/moonshinelabs-ai/zeroshot-docs/blob/main/general/getting_started.md) folder for some details on how things work under the hood.

## Get help

If you need help or just want to chat, join the [Moonshine Labs Slack server](https://join.slack.com/t/moonshinecommunity/shared_invite/zt-1rg1vnvmt-pleUR7TducaDiAhcmnqAQQ) and come hang out in the #zeroshot channel.
