# Why Use Zeroshot

Compared to other image classification solutions, Zeroshot offers a few advantages:

1. **Easy and Fast**: Unlike most classifiers, Zeroshot does not require you to provide either training data or labels. This means you can make a classifier in seconds with minimal expertise, instead of requiring weeks of development, specialized engineers, and expensive GPU hardware.

1. **Powerful**: Zeroshot has been shown to achieve near-state of the art results on a variety of standard benchmarks and tasks. For a similarly sized model, Zeroshot outperforms CLIP on many evaluation tasks.

1. **Lightweight**: Zeroshot is a lightweight library with minimal dependencies. It runs your model quickly on a CPU, and doesn't require a GPU to operate. It doesn't require PyTorch or Tensorflow or any other heavy libraries. Compared to CLIP, Zeroshot is 74% smaller and 76% faster.

1. **Free**: Zeroshot is free and open-source both to train models and use them. You can embed the model directly on your website, in your server, or on your robot. No paying a fee for every prediction you make.

1. **Offline and Privacy Preserving**: Unlike cloud-based solutions, your data never leaves your premises when you run an inference using our libraries. You can use Zeroshot without sharing your or your user's data.

### Why Not Zeroshot?

We won't pretend that there's a one size fits all solution, and there are cases where Zeroshot might not be right for you. Specifically:

1. **Custom Data Required**: Zeroshot uses an extremely large open-source dataset as the backend for training your model, but if you have custom or unique images you may not be able to find any similar items. For example, if you want to classify your new clothing line, our database will likely have never seen your different clothes. If using custom data is something that interests you, please reach out to us!

1. **You Have a Large Budget**: We've designed Zeroshot to provide a powerful model for cheap. However, if you have large amounts of labeled training data and an engineering team that's up to the task, you may be able to achieve better accuracy with a traditional supervised model (on the other hand we do find Zeroshot handily beats many fully supervised approaches).

1. **Mistakes are Costly or Safety-Critical**: Like any ML system, Zeroshot isn't perfect. If your use case requires 100% accuracy (and otherwise bad things happen), consider using a non-learned system instead.
