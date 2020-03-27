# -*- coding: utf-8 -*-

import os
from datetime import datetime

import numpy as np
import tensorflow as tf


tf.random.set_seed(42)
np.random.seed(42)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

FILTERS = 64
#
# model = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(FILTERS, 3, activation="relu", input_shape=(28, 28, 1)),
#   tf.keras.layers.MaxPooling2D(2),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.Conv2D(FILTERS, 3, activation="relu"),
#   tf.keras.layers.MaxPooling2D(2),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.Conv2D(FILTERS, 3, activation="relu"),
#   tf.keras.layers.GlobalAveragePooling2D(),
#   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Dense(10, activation="softmax")
# ])

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(FILTERS, 3, activation="relu", input_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(FILTERS, 3, activation="relu", padding="same"),
  tf.keras.layers.MaxPooling2D(2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(FILTERS, 3, activation="relu"),
  tf.keras.layers.Conv2D(FILTERS, 3, activation="relu", padding="same"),
  tf.keras.layers.MaxPooling2D(2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(FILTERS, 3, activation="relu"),
  tf.keras.layers.Conv2D(FILTERS, 3, activation="relu", padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


print(model.summary())

logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

# Define the basic TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                      update_freq="batch",
                                                      histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(logdir + "/validation")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="model.h5",
                                                 save_best_only=True)
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=3, verbose=1, mode='auto',
    restore_best_weights=True)

model.fit(train_images,
          train_labels,
          epochs=6,
          callbacks=[tensorboard_callback, cp_callback, stopping_callback],
          batch_size=128,
          validation_data=(test_images, test_labels))


tf.keras.utils.plot_model(model,
                          to_file=os.path.join(logdir, "model.png"),
                          show_shapes=True)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc, test_loss)

