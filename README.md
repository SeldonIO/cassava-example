# Deploying a Custom Tensorflow Model with MLServer and Seldon Core

## Background

### Intro

This tutorial walks through the steps required to take a python ML model from your machine to a production deployment on Kubernetes. More specifically we'll cover:
- Turning the ML model into an API
- Containerizing the model
- Storing the container in a registry
- Deploying the model to Kubernetes (with Seldon Core)
- Scaling the model

### The Use Case

For this tutorial, we're going to use the [Cassava dataset](https://www.tensorflow.org/datasets/catalog/cassava) available from the Tensorflow Catalog. This dataset includes leaf images from the cassava plant. Each plant can be classified as either "healthly" or as having one of four diseases (Mosaic Disease, Bacterial Blight, Green Mite, Brown Streak Disease).

![cassava_examples](img/cassava_examples.png)

We won't go through the steps of training the classifier. Instead, we'll be using a pre-trained one available on TensorFlow Hub. You can find the [model details here](https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2). 

## Getting Set Up

The easiest way to run this example is to clone the repository. Once you've done that, you can just run:

```Python
pip install -r requirements.txt
```

And it'll set you up with all the libraries required to run the code.

## Running The Python App

The starting point for this tutorial is python script `app.py`. This is typical of the kind of python code we'd run standalone or in a jupyter notebook. Let's familiarise ourself with the code:

```Python
from helpers import plot, preprocess
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Fixes an issue with Jax and TF competing for GPU
tf.config.experimental.set_visible_devices([], 'GPU')

# Load the model
model_path = './model'
classifier = hub.KerasLayer(model_path)

# Load the dataset and store the class names
dataset, info = tfds.load('cassava', with_info=True)
class_names = info.features['label'].names + ['unknown']

# Select a batch of examples and plot them
batch_size = 9
batch = dataset['validation'].map(preprocess).batch(batch_size).as_numpy_iterator()
examples = next(batch)
plot(examples, class_names)

# Generate predictions for the batch and plot them against their labels
predictions = classifier(examples['image'])
predictions_max = tf.argmax(predictions, axis=-1)
print(predictions_max)
plot(examples, class_names, predictions_max)
```

First up, we're importing a couple of functions from our `helpers.py` file:
- `plot` provides the visualisation of the samples, labels and predictions.
- `preprocess` is used to resize images to 224x224 pixels and normalize the RGB values.


