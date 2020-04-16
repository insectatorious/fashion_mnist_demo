import os

import pytest
import numpy as np
from PIL import Image

from vis_utils import scale_image_for_model, convert_image_to_greyscale


def test_scale_image_for_model():
  sample_image_data = np.random.randint(low=0, high=255, size=(100, 100))
  image: Image = Image.fromarray(sample_image_data, "RGB")
  resized_image: Image = scale_image_for_model(image)

  assert resized_image.size == (28, 28)

  resized_image = scale_image_for_model(image, dim=32)

  assert resized_image.size == (32, 32)


def test_convert_image_to_greyscale():
  sample_image_data = np.random.randint(low=0, high=255, size=(100, 100, 3))
  image: Image = Image.fromarray(sample_image_data, "RGB")
  greyscale_image = convert_image_to_greyscale(image)

  assert len(greyscale_image.getbands()) == 1
