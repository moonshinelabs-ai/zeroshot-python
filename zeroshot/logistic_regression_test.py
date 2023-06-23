import unittest

import numpy as np

from .logistic_regression import LogisticRegression, softmax


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.logistic_regression = LogisticRegression(
            coefs=np.array([[0.5, -0.5], [0.2, 0.7]]), intercept=np.array([0.1, -0.1])
        )
        self.X = np.array([[0.5, 0.3], [0.7, 0.1], [0.2, 0.6]])

    def test_softmax(self):
        to_test = np.array([[0, 1], [1, 0], [0.2, 0.2], [1, 15]])
        expected_result = np.array(
            [
                [2.68941421e-01, 7.31058579e-01],
                [7.31058579e-01, 2.68941421e-01],
                [5.00000000e-01, 5.00000000e-01],
                [8.31528028e-07, 9.99999168e-01],
            ]
        )

        result = softmax(to_test)
        np.testing.assert_almost_equal(result, expected_result, decimal=7)

    def test_predict_proba(self):
        result = self.logistic_regression.predict_proba(self.X)
        expected_result = np.array(
            [
                [0.49750002, 0.50249998],
                [0.57199613, 0.42800387],
                [0.38698582, 0.61301418],
            ]
        )
        np.testing.assert_almost_equal(result, expected_result, decimal=7)

    def test_predict(self):
        result = self.logistic_regression.predict(self.X)
        expected_result = np.array([1, 0, 1])
        np.testing.assert_array_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
