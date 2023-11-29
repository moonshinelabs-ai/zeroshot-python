import io
import unittest

import numpy as np
from PIL import Image

from .utils import _from_data_or_path


class TestImageLoad(unittest.TestCase):
    def test_all_red_image(self):
        # Create an all-red image of size 10x10
        red_image = Image.new("RGB", (10, 10), "red")

        # Save to a BytesIO object
        img_byte_arr = io.BytesIO()
        red_image.save(img_byte_arr, format="PNG")
        img_byte_arr = io.BytesIO(img_byte_arr.getvalue())

        # Call the function
        result = _from_data_or_path(img_byte_arr)

        # Create the expected output
        expected_output = np.array([[[255, 0, 0]] * 10] * 10)

        # Assert the numpy array is as expected
        np.testing.assert_array_equal(result, expected_output)
