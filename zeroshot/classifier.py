import numpy as np

class Classifier(object):
    def __init__(self, path: Optional[str] = None, model_id: Optional[str] = None):
        self.path = path
        self.model_id = model_id

    def predict(self, image: Union[str, np.ndarray]) -> str:
        """Predicts the class of an image.

        Args:
            image (Union[str, np.ndarray]): The image to predict, either a url or a numpy array.

        Returns:
            np.ndarray: The predicted class
        """
        raise NotImplementedError
