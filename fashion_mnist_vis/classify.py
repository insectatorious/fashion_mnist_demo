# -*- coding: utf-8 -*-

import os
import logging
import argparse
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from vis_utils import convert_image_to_greyscale, valid_keras_model
from vis_utils import scale_image_for_model
from vis_utils import class_names
from vis_utils import visualise_feature_maps
from vis_utils import save_feature_map
from vis_utils import get_cam, plot_cam

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def main(FLAGS) -> None:
  model: tf.keras.Model = FLAGS.saved_model
  greyscale_image: Image = convert_image_to_greyscale(FLAGS.test_image)
  rescaled_image: Image = scale_image_for_model(greyscale_image)

  model_input: np.ndarray = np.array(rescaled_image)[tf.newaxis, ..., tf.newaxis] / 255.0
  pred: np.ndarray = model.predict(model_input)
  logging.debug(f"Predictions: {pred}")

  pred_class: str = class_names[np.argmax(pred)]
  score: float = np.max(pred)

  logging.info(f"Classification is '{pred_class}' with a score of {score}")

  if FLAGS.save_plots:
    logging.info("Visualising classification, saving to 'visualisations' dir")
    os.makedirs("visualisations", exist_ok=True)

    vis_model = tf.keras.models.Model(
      inputs=model.input,
      outputs=[layer.output for layer in model.layers])

    feature_maps = vis_model.predict(model_input)
    layer_names_with_index: List[Tuple[int, str]] = [
      (index, layer.name)
      for index, layer in enumerate(model.layers)
      if "dropout" not in layer.name]

    for i, layer_name in layer_names_with_index:
      save_feature_map(fig=visualise_feature_maps(feature_maps[i],
                                                  layer_name),
                       output_dir=FLAGS.plot_dir,
                       fname=f"{layer_name}.png")

    # fig = visualise_feature_maps(feature_maps[1], layer_names_with_index[1][1])
    # fig.savefig(fname=os.path.join(FLAGS.plot_dir, "fmap1.png"))

    cam_layer_with_index: Tuple[int, str] = [(i, name)
                 for i, name in layer_names_with_index
                 if "global_average_pooling" in name or "flatten" in name][-1]
    if cam_layer_with_index:
      logging.info(f"Class Activation Map Layer: {cam_layer_with_index[1]}")
      # We want the input to the layer
      cam_layer_with_index = cam_layer_with_index[0] - 1, cam_layer_with_index[1]

      cam = get_cam(image_size=model_input.shape[1],
                    conv_out=feature_maps[cam_layer_with_index[0]],
                    pred_vec=pred,
                    all_amp_layer_weights=model.layers[-1].get_weights()[0],
                    filters=model.layers[cam_layer_with_index[0]].output.shape[-1])
      fig = plot_cam(model_input, cam)
      save_feature_map(fig, output_dir=FLAGS.plot_dir, fname=f"cam.png")

    FLAGS.test_image.save(os.path.join(FLAGS.plot_dir, "input_image.png"))
    greyscale_image.save(os.path.join(FLAGS.plot_dir, "greyscale_input.png"))
    rescaled_image.save(os.path.join(FLAGS.plot_dir, "rescaled_model_input.png"))


def valid_test_image(value: str) -> Image:
  if not os.path.exists(value):
    raise argparse.ArgumentTypeError(f"Expected 'test_image' to be a valid "
                                     f"path, got '{value}'")

  logging.info(f"Loading test image from '{value}'")

  return Image.open(value)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "test_image",
    default="",
    type=valid_test_image,
    help="Path to image need classification (will be resized to 28x28)"
  )
  parser.add_argument(
    "--saved_model",
    default="model.h5",
    type=valid_keras_model,
    help="Path to a saved TF Keras model")
  parser.add_argument("--save_plots",
                      help="Save visualisations of this classification to "
                           "'visualisations' directory",
                      action="store_true")
  parser.add_argument(
    "--plot_dir",
    default="visualisations",
    type=str,
    help="Path to save visualisations to (dir)")

  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    logging.warning("Unparsed arguments: {}".format(unparsed))

  logging.info("Arguments: {}".format(FLAGS))
  main(FLAGS)
