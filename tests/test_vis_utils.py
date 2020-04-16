# -*- coding: utf-8 -*-
import argparse
import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import matplotlib.pyplot as plt
from PIL import Image

from vis_utils import scale_image_for_model, valid_keras_model, save_feature_map
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


def test_create_master_sprite():
  # 9 colour images of 100x100 resolution
  sample_image_data = np.random.randint(low=0, high=255, size=(9, 100, 100, 3))
  master_sprite_name: str = "master.jpg"

  with TemporaryDirectory() as tmpdir:
    create_master_sprite(sample_image_data, tmpdir)
    assert os.path.exists(os.path.join(tmpdir, master_sprite_name)), \
      (f"Expected master sprite as '{master_sprite_name}' in {tmpdir} which "
       f"has {os.listdir(tmpdir)}")


def test_valid_keras_model():
  model_name: str = "model.h5"
  if not os.path.exists(model_name):
    model_name = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              model_name)
  model = valid_keras_model(model_name)
  assert model


def test_valid_kears_model_invalid_path():
  model_name: str = "banana.h5"
  with pytest.raises(argparse.ArgumentTypeError):
    valid_keras_model(model_name)


def test_save_feature_map():
  fig = plt.figure()
  filename = "banana.jpg"
  with TemporaryDirectory() as tmpdir:
    save_feature_map(fig=fig, output_dir=tmpdir, fname=filename)
    assert os.path.exists(os.path.join(tmpdir, filename)), \
      (f"Expected '{filename}' to be in '{tmpdir}' which has "
       f"'{os.listdir(tmpdir)}'")

