# -*- coding: utf-8 -*-

import io
import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from vis_utils import valid_keras_model, create_master_sprite, class_names

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def main(FLAGS):
  dir_path: str = "tensorboard_assets"
  num_to_export: int = 10000

  if not os.path.exists(dir_path):
    logging.warning(f"Expected to find '{dir_path}' directory, creating. "
                    f"Remember to copy 'config.json' from "
                    f"project_root/tensorboard_assets/config.json")
    os.makedirs(dir_path)

  model: tf.keras.Model = FLAGS.saved_model
  cam_model: tf.keras.Model = tf.keras.models.Model(inputs=model.input,
                                    outputs=[model.layers[-3].output,
                                             model.layers[-1].output])
  fashion_mnist = tf.keras.datasets.fashion_mnist

  (_, _), (test_images, test_labels) = fashion_mnist.load_data()

  create_master_sprite(test_images[:num_to_export], dir_path)

  test_images = test_images / 255.0
  test_images = test_images[..., tf.newaxis]

  test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
  logging.info(f"Visualising model with test_loss: {np.round(test_loss, 4)}, "
               f"and test_acc: {np.round(test_acc, 4)}")

  embedding_vec, pred_vec = cam_model.predict(test_images[:num_to_export])
  pred_class = [class_names[cls] for cls in np.argmax(pred_vec, axis=1)]
  true_class = [class_names[cls] for cls in test_labels[:num_to_export]]
  scores = np.max(pred_vec, axis=1)
  flattened_vec: np.ndarray = test_images.reshape(
    test_images.shape[0],
    test_images.shape[1] * test_images.shape[2])

  np.savetxt(os.path.join(dir_path, "embedding_tensors.tsv"),
             embedding_vec,
             delimiter='\t',
             encoding="utf-8")
  np.savetxt(os.path.join(dir_path, "raw_tensors.tsv"),
             flattened_vec[:num_to_export],
             delimiter='\t',
             encoding="utf-8")

  with io.open(os.path.join(dir_path, "metadata.tsv"), "w", encoding='utf-8') as m:
    m.write("\t".join(["Label", "Predicted", "Score"]) + "\n")
    for label, pred, score in zip(true_class, pred_class, scores):
      m.write("\t".join([str(label), str(pred), str(score)]) + "\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--saved_model",
    default="model.h5",
    type=valid_keras_model,
    help="Path to a saved TF Keras model")

  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    logging.warning("Unparsed arguments: {}".format(unparsed))

  logging.info("Arguments: {}".format(FLAGS))
  main(FLAGS)
