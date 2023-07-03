import numpy as np


def softmax(X: np.ndarray) -> np.ndarray:
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def expit(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))


class LogisticRegression(object):
    def __init__(self, coefs: np.ndarray, intercept: np.ndarray):
        self.coefs = coefs
        self.intercept = intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # For a binary classifier, we need to add the negative of the coefficients.
        z = np.dot(X, self.coefs.T) + self.intercept
        if self.coefs.shape[0] == 1:
            prob = expit(z)
            return np.vstack([1 - prob, prob]).T
        else:
            return softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
