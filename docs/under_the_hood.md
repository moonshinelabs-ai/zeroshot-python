# Under the Hood

How does it work?

Zeroshot uses a combination of two powerful AI models to create classifiers from text. The major components are:

- A large image dataset (LAION-5B) and a text search model (CLIP).
- A powerful but small pre-trained model (DinoV2).
- A simple web-app that assembles the final model.

So what exactly is happening when you create a model on [usezeroshot.com](https://www.usezeroshot.com)?

1. We convert your text or image into an embedding using [CLIP-L/14](https://openai.com/research/clip), a powerful text-image contrastive model.
1. We use this embedding to search the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset and find a hundred or so images that most closely match.
1. For all of the images our search returns, we pass them through the [DINOv2-S/14](https://dinov2.metademolab.com/) pre-trained model to create a feature vector.
1. Once you've done this for all of your classes, we train a simple logistic regression model using the feature vectors we got in step three.
1. When you use the client library, we process your image using the same DINO backbone, and then run the output through the logistic regression model to get the final prediction.
