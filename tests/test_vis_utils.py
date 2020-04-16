import os
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from vis_utils import scale_image_for_model
from vis_utils import convert_image_to_greyscale
from vis_utils import create_sprite
from vis_utils import create_master_sprite


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


def test_create_sprite_greyscale_image():
  # 10 greyscale images of 100x100 resolution
  sample_image_data = np.random.randint(low=0, high=255, size=(10, 100, 100))
  create_sprite_ndarray_assertions(sample_image_data)


def test_create_sprite_rgb_image():
  # 10 colour images of 100x100 resolution
  sample_image_data = np.random.randint(low=0, high=255, size=(10, 100, 100, 3))
  create_sprite_ndarray_assertions(sample_image_data)


def create_sprite_ndarray_assertions(sample_image_data):
  sprite: np.ndarray = create_sprite(sample_image_data)
  assert len(sprite.shape) == 3, (f"Expected sprite shape to be Rank 3 tensor,"
                                  f"shape is {sprite.shape}")
  assert sprite.shape[-1] == 3, (f"Expected sprite image to be RGB (3 channels),"
                                 f" shape is {sprite.shape}")
  assert sprite.shape[0] == sprite.shape[1], (f"Expected sprite image to be "
                                              f"square, got {sprite.shape}")
