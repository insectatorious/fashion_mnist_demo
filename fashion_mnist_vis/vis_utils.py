# -*- coding: utf-8 -*-

import os
import logging
import argparse
import itertools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Colormap
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import ImageGrid


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_confusion_matrix(cm, class_names):
  """Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
  ----
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  return figure


def get_cam(image_size, conv_out, pred_vec, all_amp_layer_weights, filters):
  conv_out = np.squeeze(conv_out)
  zoom_scale = image_size / conv_out.shape[0]
  mat_for_mult = zoom(conv_out, (zoom_scale, zoom_scale, 1), order=1)
  pred = np.argmax(pred_vec)
  amp_layer_weights = all_amp_layer_weights[:, pred]

  final_output = np.dot(mat_for_mult.reshape((image_size * image_size, filters)),
                        amp_layer_weights).reshape(image_size, image_size)

  return final_output


def process_image(ax, img, all_amp_layer_weights, conv_out_model, class_names, filters):
  conv_out, pred_vec = conv_out_model.predict(img[np.newaxis, ...])
  # 'img' 1st dimension (shape[0]) is assumed to be batch size ☝
  cam = get_cam(img.shape[1], conv_out, pred_vec, all_amp_layer_weights, filters)

  ax.imshow(np.squeeze(img), cmap=plt.cm.binary)
  ax.imshow(cam, cmap=plt.cm.jet, alpha=0.5)
  ax.set_title(f"ŷ: {class_names[np.argmax(pred_vec)]}")


def visualise_feature_maps(
    layer_feature_maps: np.ndarray,
    layer_name: str,
    figure_size: int = 14,
    colour_map: Colormap = plt.cm.get_cmap("viridis")) -> plt.Figure:
  fig = plt.figure(figsize=(figure_size, figure_size))
  # fig.suptitle(f"Layer: {layer_name}", fontsize=16)

  if len(layer_feature_maps.shape) == 4:
    #  We have a 2D feature map - need to show a grid
    nrows: int = int(np.ceil(np.sqrt(layer_feature_maps.shape[-1])))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     axes_pad=0.3,  # pad between axes in inch.
                     nrows_ncols=(nrows, nrows))

    for ax, f in zip(grid, np.moveaxis(layer_feature_maps, 3, 1)[0]):
      ax.imshow(f, cmap=colour_map)

    fig.tight_layout()

  elif len(layer_feature_maps.shape) == 2:
    # We have a 1D feature map - need to show a single image
    fig = plt.figure(figsize=(figure_size, 1))
    plt.imshow(layer_feature_maps)
    plt.axis("off")
    plt.tight_layout()

  else:
    logging.warning(f"Expected 'layer_feature_maps' to have 4D or 2D shape, "
                    f"got {layer_feature_maps.shape}")

  return fig


def valid_keras_model(value: str) -> tf.keras.Model:
  if not os.path.exists(value):
    raise argparse.ArgumentTypeError(f"Expected 'saved_model' to be a valid "
                                     f"path, got '{value}'")
  logging.info(f"Loading Keras model from '{value}'")

  return tf.keras.models.load_model(value)


def save_feature_map(fig: plt.Figure, output_dir: str, fname: str) -> None:
  fig.savefig(fname=os.path.join(output_dir, fname))


def plot_cam(img: np.ndarray,
             cam: np.ndarray,
             figure_size: int = 14) -> plt.Figure:
  fig = plt.figure(figsize=(figure_size, figure_size))
  # fig.suptitle(fname, fontsize=16)
  plt.axis("off")

  plt.imshow(np.squeeze(img), cmap=plt.cm.binary)
  plt.imshow(cam, cmap=plt.cm.jet, alpha=0.5)

  return fig


def convert_image_to_greyscale(image: Image) -> Image:
  """Converts an Image to greyscale.

  See https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.convert

  Args:
  ----
    image: PIL.Image

  Returns: PIL.Image

  """
  return image.convert('L')


def scale_image_for_model(image: Image, dim: int = 28) -> Image:
  return image.resize(size=(dim, dim), resample=Image.BICUBIC)


def create_sprite(data: np.ndarray) -> np.ndarray:

  # For B&W or greyscale images
  if len(data.shape) == 3:
    data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

  n = int(np.ceil(np.sqrt(data.shape[0])))
  n = np.min([n, 8192])
  padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
  data = np.pad(data, padding, mode='constant',
                constant_values=0)

  # Tile images into sprite
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
  # print(data.shape) => (n, image_height, n, image_width, 3)

  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  # print(data.shape) => (n * image_height, n * image_width, 3)
  return data


def create_master_sprite(images: np.ndarray, output_dir: str) -> None:
  sprite = create_sprite(images)
  sprite = Image.fromarray(sprite, "RGB")
  sprite.save(os.path.join(output_dir, "master.jpg"))
