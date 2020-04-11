import logging
import argparse
import time

import cv2
import numpy as np
import tensorflow as tf

from vis_utils import valid_keras_model, get_cam, class_names

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def main(FLAGS):
  model: tf.keras.Model = FLAGS.saved_model
  cam_model = tf.keras.models.Model(inputs=model.input,
                                    outputs=[model.layers[-4].output,
                                             model.layers[-1].output])
  filters = model.layers[-4].output.shape[-1]
  all_weights = model.layers[-1].get_weights()[0]
  cap = cv2.VideoCapture(0)
  alpha = 0.4
  flag = True
  while (True):
    # Capture frame-by-frame
    _, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # COLOR_BGR2GRAY, COLOR_BGR2RGB
    output = img.copy()
    img_small = cv2.resize(img, (28, 28))
    conv_out, pred_vec = cam_model.predict(img_small[np.newaxis, ..., np.newaxis])

    if flag:
      cam = get_cam(image_size=img_small.shape[1],
                    conv_out=conv_out,
                    pred_vec=pred_vec,
                    all_amp_layer_weights=all_weights,
                    filters=filters)


      cam_large = cv2.resize(cam, (img.shape[1], img.shape[0]))


    # print(f"Classification: {class_names[np.argmax(pred_vec)]} ({np.round(np.max(pred_vec, 4))})")
    cv2.putText(output, f"Classification: {class_names[np.argmax(pred_vec)]} "
                        f"({np.round(np.max(pred_vec), 4)})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0xff, 3)

    cv2.addWeighted(cam_large, alpha, output, 1 - alpha, 0., output, dtype=cv2.CV_8U)

    cv2.imshow('frame', output)
    # object_keypoints = feature_detector(img, detector)
    # cv2.imshow('frame', object_keypoints)

    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
      break
    elif k == -1:  # normally -1 returned,so don't print it
      continue
    elif k == 49:  # 1
      pass
    elif k == 50:  # 2
      pass
    elif k == 51:  # 3
      pass
    else:
      print(k)  # Dont know what to do so print it

    time.sleep(1.5)

  cap.release()
  cv2.destroyAllWindows()


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
