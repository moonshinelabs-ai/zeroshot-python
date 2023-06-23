import numpy as np


def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


class LogisticRegression(object):
    def __init__(self, coefs: np.ndarray, intercept: np.ndarray):
        self.coefs = coefs
        self.intercept = intercept

    def predict_proba(self, X):
        z = np.dot(X, self.coefs.T) + self.intercept
        probabilities = softmax(z)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
