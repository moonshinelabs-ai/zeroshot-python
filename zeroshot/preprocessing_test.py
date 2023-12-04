import unittest

import numpy as np

from .preprocessing import create_preprocess_fn


class TestPreprocessFunctions(unittest.TestCase):
    def test_standard_preprocess_returns_correct_shape_and_range(self):
        preprocess_fn = create_preprocess_fn("dino")
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed_image = preprocess_fn(sample_image)

        self.assertEqual(processed_image.shape, (224, 224, 3))
        self.assertEqual(processed_image.dtype, np.float32)
        self.assertLessEqual(np.mean(processed_image), 1)

    def test_variable_preprocess_returns_correct_shape_and_range(self):
        preprocess_fn = create_preprocess_fn("dino_variable")
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed_image = preprocess_fn(sample_image)

        self.assertEqual(processed_image.shape, (224, 224, 3))
        self.assertEqual(processed_image.dtype, np.float32)
        self.assertLessEqual(np.mean(processed_image), 1)

    def test_variable_preprocess_reduces_shape_to_patch_size(self):
        preprocess_fn = create_preprocess_fn("dino_variable")
        sample_image = np.random.randint(0, 255, (228, 251, 3), dtype=np.uint8)
        processed_image = preprocess_fn(sample_image)

        self.assertEqual(processed_image.shape[0] % 14, 0)
        self.assertEqual(processed_image.shape[1] % 14, 0)

    def test_invalid_preprocess_name_raises(self):
        with self.assertRaises(ValueError):
            create_preprocess_fn("invalid_name")


# This allows running the tests from the command line
if __name__ == "__main__":
    unittest.main()
