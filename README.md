# Fashion MNIST Image Classification & Visualisation

This repository contains sample code to train and visualise a simple [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) to classify the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using [TensorFlow 2.x](https://www.tensorflow.org).
Whilst the network architecture is a simple [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) model ([details below](#model-structure)), the goal is to highlight the ability to visualise the model as it classifies input images. 

![tsne_gif](docs/images/tsne.gif)
👉 [**Live demo**](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/insectatorious/fashion_mnist_demo/master/tensorboard_assets/config_github.json) of embedding vectors on Tensorboard.

# Table Of Contents

- [Summary](#summary)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Training](#training)
  - [Classifying New Images](#classifying-new-images)
  - [Exporting Tensorboard Assets](#exporting-tensorboard-assets)
- [Model Structure](#model-structure)
- [Network Layer Visualisations](#network-layer-visualisations)
  - [Transformations To Input Image](#transformations-to-input-image)
  - [Class Activation Map](#class-activation-map)
  - [Layer Activations](#layer-activations)
  - [Global Average Pooling & Dense Activations](#global-average-pooling--dense-activations)
- [Embedding Vectors](#embedding-vectors)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [t-Distributed Stochastic Neighbour Embedding (t-SNE)](#t-distributed-stochastic-neighbour-embedding-t-sne)
  - [Custom Projections](#custom-projections-tensorboard-only)
- [Live Demo](#play-with-the-vectors-yourself)
- [Further Reading](#further-reading)
- [Licence](#licence)

# Summary

- A Convolutional Neural Network is trained (see [train.py](./fashion_mnist_vis/train.py)) to classify images from the Fashion MNIST dataset. 
  - Model checkpoints save on validation loss improvements
  - Early Stopping prevents runaway training. 
  - Dropout to limit overfitting
  - Global Average Pooling to simplify feature extraction along with spatial invariance
  - Tensorboard logging of Images, Histograms and Distributions along with Scalars like `accuracy` & `loss`
- Classification of a new image (passed in using a command line argument or one of the sample images located in [sample_images](./sample_images)) using a trained model (:point_up:) using [classify.py](./fashion_mnist_vis/classify.py).
- Exporting test images from dataset for visualisation in Tensorboard Projector

Details about the dataset can be found [here](https://github.com/zalandoresearch/fashion-mnist). Briefly, each image is `28x28` pixels and is one of ten different types of fashion categories (Shirt, Dress, Sneakers etc). The classification task is to train a model that can take one of these images as input and classify it into one of the existing categories. After training several visualisations are generated to see the model's learning. 

These visualisations cover:
- [Transformations to input image](#transformations-to-input-image): transforming the input image before it is passed as input to the network
- [Layer activations](#layer-activations): output of each feature map of each layer in the network for a single input
- [Class activation map](#class-activation-map): a heatmap overlaid on the input image to see where the network is paying 'Attention'
- [Live Demo](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/insectatorious/fashion_mnist_demo/master/tensorboard_assets/config_github.json): exporting embedded vectors for each input in the test set for visualistion and analysis in Tensorboard Projector

# Usage

## Requirements

A full list of requirements can be found in [`requirements.txt`](./requirements.txt). Additionally `graphviz` is required for plotting the model's structure (this is an OS level install). The main dependencies of note are:
- TensorFlow 2.x
- pillow (used for image loading, greyscale conversions and resizing)
- opencv (used for running model against webcam feed - highly experimental)
- pydot & graphviz (used for plotting model structure)

All dependecies can be installed by running (note `graphviz` needs to be installed on the system):
```bash
# execute from project root folder
pip install -r requirements.txt
```

## Training

Simply run (from the project root path):

```bash
PYTHONPATH=fashion_mnist_vis python fashion_mnist_vis/train.py
```
The trained model will be saved in the project root with the filename `model.h5`. 

## Classifying New Images

To classify one of the sample images:

```bash
PYTHONPATH=fashion_mnist_vis python fashion_mnist_vis/classify.py sample_images/black_bag.jpg --saved_model model.h5 --save_plots
```

Here the saved model (`model.h5`) and image (`black_bag.jpg`) can be substituted as needed. All visualisations will be stored  in the `visualisations` folder. To avoid plotting each time and only classify the image, remove the `--save_plots` argument form the command.

## Exporting Tensorboard Assets


# Model structure

Summary | Graph Diagram
----|----
A simple, sequential Convolutional Neural Network with [Batch Normalisation](https://en.wikipedia.org/wiki/Batch_normalization), a [Global Average Pooling](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/) layer (for [Attention](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)) and [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout). Note there are no fully connected layers in this network. This model achieved an accuracy of 92.07% on the test set. It is expected that a model with skip connections as popularised by the [ResNet-50](https://arxiv.org/abs/1512.03385) architecture would improve the classification capabilities. However the visualisation of such a network with merge layers would be more difficult and harder to grok for people starting out. | <img src="docs/images/model.png" width="500">

# Network Layer Visualisations

### Input Image
A black handbag from Argos is chosen as our input - [KIPLING Black Art Mini Handbag - One Size](https://www.argos.co.uk/product/tuc135958689)

![Input Image](docs/images/visualisations/input_image.png)

Model classification: `Bag` with a score of `0.9213`.

Image is from an Argos product page so out of the train and test datasets. This is a crucial validation of the model's ability to generalise and work on data that's not restricted to the dataset itself. CNNs have tremendous memorisation capabilities and seeing its ability to correctly classif data that is from a different source (but still in line with the training dataset's patterns) is fundamental to proving its effectiveness.

### Transformations To Input Image
The model only accepts greyscale images with a resolution of `28 x 28` so all input images will need to be 
- converted to greyscale
- resized to `28 x 28`

before they can be sent to the model.

### Class Activation Map

Input To Model | Class Activation Map 
---------------|---------------------
<img src="docs/images/visualisations/rescaled_model_input.png" width="100"> | <img src="docs/images/visualisations/cam.png" width="100"> 

Looking at the activation map :thinking:, it appears the model is paying attention to the *handle of the bag* in making it's classification (along with the absence of anything above the handle).

### Layer Activations
The transformed image (as detailed above) passes through the network and each of the feature maps in each layers extracts some features from it. The lower layers of the network (CNN Layer 1 & 2 below :point_down:) typically end up as edge detectors. Specifically they look for certain kinds of edges that are of 'use' to the layers deeper in the network. Layers futher down in the network use these features to activate when certain criteria is met. For example, the first few layers of feature maps might activate on a pair of curved edges near the top middle of the image (like seen in the handle of a bag. Higher layers will then activate when seeing these features to indicate that there is strong probability that a bag's handle is visible in the image. Eventually the final few layers will activate to indicate a 'Bag' class if all the collection of features most closely match a bag (a handle, a solid shape in the middle etc). 

#### CNN Layer Activations

| CNN Layer 1 | CNN Layer 2 |
|   :---:   |   :---:   |
| <img src="docs/images/visualisations/conv2d.png" height="500" width="500"> | <img src="docs/images/visualisations/conv2d_1.png" height="500" width="500"> |

:point_up: We see 64 feature maps in the two layers above showing different activations for the bag. Invariably, some of these will be blank as they only activate when detecting edges of other classes (like 'Ankle Boot' or 'Sneaker'). 

:point_down: The last few convolutional layers (5 & 6) do not bear any recognisable resemblance to the input image, however they are showing activations on groups of features. 

| CNN Layer 5 | CNN Layer 6 |
|   :---:   |   :---:   |
| <img src="docs/images/visualisations/conv2d_4.png" height="500" width="500"> | <img src="docs/images/visualisations/conv2d_5.png" height="500" width="500"> |

### Global Average Pooling & Dense Activations

The activations from the previous layer are averaged using a [Global Average Pooling](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/) layer. Activations from this layer provide the *embedding vector* (see next section) that the model uses to make the final classification. 

| Global Average Pooling | Dense (Final) |
|   :---:   |   :---:   |
| <img src="docs/images/visualisations/global_average_pooling2d.png" height="100" width="500"> | <img src="docs/images/visualisations/dense.png" height="50" width="500"> |

Note: the Dropout layer is not visualised as it is only used whilst training the network. When making a prediction the network does not perform any function. 

# Embedding Vectors
An [embedding vector](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) provides a low-dimensional representation of an input datapoint. In the case of this model each input datapoint can be thought of a `28 x 28 = 784` dimensional vector. As this image passes through the network's layers, it is transformed until it is ultimately a `64` dimension vector. This is the penultimate layer in the network and contains enough information about the datapoint to allow the final layer to perform the classification. 

The following visualisations are aimed to examining these vectors to understand how the model is 'representing' the images. As we cannot visualise a 64 dimension vector directly, we have to perform some kind of dimensionality reduction to get it down to two or three dimensions.

## Principal Component Analysis (PCA)
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is a simple, powerful algorithm to reduce a high dimensional vector to low dimenions. In the images below, the `64` dimension embedding vector is reduced to 3 dimensions and plotted by Tensorboard. A quick and easy starting point for analysing high dimensional data it does struggle to deal with non-linearity in the higher dimensions. 

Image 1 | Image 2
----|----
<img src="docs/images/pca_1.png" width="500"> | <img src="docs/images/pca_2.png" width="500"> 


## t-Distributed Stochastic Neighbour Embedding (t-SNE)
[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) is a non-deterministic algorithm to visualise high dimenional data in lower dimensions whilst retaining spatial information. See [this excellent article on how to effectively use t-SNE](https://distill.pub/2016/misread-tsne/). It's important to note that unlike PCA, t-SNE is not a generalised function that takes high dimensional data and output low dimensional equivalents. Instead, it provides a mapping for known datapoints only. 

Image 1 | Image 2
---|---
<img src="docs/images/tsne_1.png" width="500"> | <img src="docs/images/tnse_2.png" width="500"> 

## Custom Projections (Tensorboard Only)
Tensorboard allows users to map projections for datapoints matching certain criteria against each other. 
In the following examples datapoints matching the 'Sneaker' vectors are projected to the left and those matching 'Ankle' are projected to the right. This can be interpreted as, the more to the left or right a data point is, the more similar it is to a 'Sneaker' or 'Ankle' datapoint. Similarly, 'Bag' and 'Coat' are projected up and down respectively. 

![Image1](docs/images/custom_1.png)

The image below :point_down: shows an example of 'Sandal' images highlighted. Most of them are closer to the right making them similar to 'Ankle Boot' but a number of them are also seen on the left making these similar to 'Sneakers'.

![Image2](docs/images/custom_2.png)

# Play With The Vectors Yourself!
View the embedded vectors on [Tensorboard](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/insectatorious/fashion_mnist_demo/master/tensorboard_assets/config_github.json). Works best in Chrome or Firefox. 

The vectors visualised here are exported using the model `model.h5`, saved in this repository. 

# Further Reading

- [Good Overview of Convolutional Neural Networks](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)
- [Conv Nets: A Modular Perspective](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/)
- [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)
- [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
- [Transfer Learning With A Pretrained ConvNet](https://www.tensorflow.org/tutorials/images/transfer_learning)

# Licence
GNU General Public License v3.0
