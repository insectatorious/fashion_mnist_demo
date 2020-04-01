# Fashion MNIST Image Classification & Visualisation

This repository contains sample code to train and visualise a simple [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) to classify the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using [TensorFlow 2.x](https://www.tensorflow.org).
Whilst the network architecture is a simple [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) model, the goal is to highlight the ability to visualise the model as it classifies input images. 

![tsne_gif](docs/images/tsne.gif)
👉 [Live demo](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/insectatorious/fashion_mnist_demo/master/tensorboard_assets/config_github.json) of embedding vectors on Tensorboard.

These visualisations cover:
- seeing the transformation of the input image before it is passed as input to the network
- seeing the output of each feature map of each layer in the network for a given input image
- seeing a Class Activation Map overlaid on the input image to see where the network is paying 'Attention'
- exporting embedded vectors for each input in the test test for visualistion and analysis in Tensorboard Projector

# CNN Layer Feature Map Activations

### Input Image
![Input Image](docs/images/visualisations/input_image.png)

Model classification: `Bag` with a score of `0.9213`.

Image is from an Argos product page so out of the train and test datasets. 

### Class Activation Map
![Class activation map](docs/images/visualisations/cam.png)
### CNN Layer 1
![CNN Activations](docs/images/visualisations/conv2d.png)


## Model structure

A simple, sequential convolutional neural network with Batch Normalisation, a Global Average Pooling layer (for Attention) and Dropout.
This model achieved an accuracy of 92.07% on the test set.
![model structure](docs/images/model.png)

It is expected that a model with skip connections as popularised by the ResNet-50 architecture would improve the classification capabilities.

## Play with the vectors yourself!
View the embedded vectors on [Tensorboard](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/insectatorious/fashion_mnist_demo/master/tensorboard_assets/config_github.json). Works best in Chrome or Firefox. 

# Licence
GNU General Public License v3.0
