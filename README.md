# Deploying a Custom Tensorflow Model with MLServer and Seldon Core

## Background

### Intro

This tutorial walks through the steps required to take a python ML model from your machine to a production deployment on Kubernetes. More specifically we'll cover:
- Running the model locally
- Turning the ML model into an API
- Containerizing the model
- Storing the container in a registry
- Deploying the model to Kubernetes (with Seldon Core)
- Scaling the model

The tutorial comes with an accompanying video which you might find useful as you work through the steps:
[![video_play_icon](img/video_play.png)](https://youtu.be/3bR25_qpokM)

The slides used in the video can be found [here](img/slides.pdf).

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

The rest of the code is fairly self-explanatory from the comments. We load the model and dataset, select some examples, make predictions and then plot the results.

Try it yourself by running:

```Bash
python app.py
```

Here's what our setup currently looks like:
![step_1](img/step_1.png)

## Creating an API for The Model

The problem with running our code like we did earlier is that it's not accessible to anyone who doesn't have the python script (and all of it's dependencies). A good way to solve this is to turn our model into an API. 

Typically people turn to popular python web servers like [Flask](https://github.com/pallets/flask) or [FastAPI](https://github.com/tiangolo/fastapi). This is a good approach and gives you lots of flexibility but it also requires you to do a lot of the work yourself. You need to impelement routes, set up logging, capture metrics and define an API schema among other things. A simpler way to tackle this problem is to use an inference server. For this tutorial we're going to use the open source [MLServer](https://github.com/SeldonIO/MLServer) framework. 

MLServer supports a bunch of [inference runtimes](https://mlserver.readthedocs.io/en/stable/runtimes/index.html) out of the box, but it also supports [custom python code](https://mlserver.readthedocs.io/en/stable/user-guide/custom.html) which is what we'll use for our Tensorflow model.

### Setting Things Up

In order to get our model ready to run on MLServer we need to wrap it in a single python class with two methods, `load()` and `predict()`. Let's take a look at the code (found in `model/serve-model.py`):

```Python
from mlserver import MLModel
from mlserver.codecs import decode_args
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Define a class for our Model, inheriting the MLModel class from MLServer
class CassavaModel(MLModel):

  # Load the model into memory
  async def load(self) -> bool:
    tf.config.experimental.set_visible_devices([], 'GPU')
    model_path = '.'
    self._model = hub.KerasLayer(model_path)
    self.ready = True
    return self.ready

  # Logic for making predictions against our model
  @decode_args
  async def predict(self, payload: np.ndarray) -> np.ndarray:
    # convert payload to tf.tensor
    payload_tensor = tf.constant(payload)

    # Make predictions
    predictions = self._model(payload_tensor)
    predictions_max = tf.argmax(predictions, axis=-1)

    # convert predictions to np.ndarray
    response_data = np.array(predictions_max)

    return response_data
```

The `load()` method is used to define any logic required to set up your model for inference. In our case, we're loading the model weights into `self._model`. The `predict()` method is where we include all of our prediction logic. 

You'll notice that we've slightly modified our code from earlier (in `app.py`). The biggest change is that it is now wrapped in a single class `CassavaModel`.

The only other task we need to do to run our model on MLServer is to specify a `model-settings.json` file:

```Json
{
    "name": "cassava",
    "implementation": "serve-model.CassavaModel"
}
```

This is a simple configuration file that tells MLServer how to handle our model. In our case, we've provided a name for our model and told MLServer where to look for our model class (`serve-model.CassavaModel`).

### Serving The Model

We're now ready to serve our model with MLServer. If you navigate to the `model/` directory using:

```bash
cd model/
```

Then you can simply run:

```bash
mlserver start model/
```

MLServer will now start up, load our cassava model and provide access through both a REST and gRPC API.

### Making Predictions Using The API

Now that our API is up and running. Open a new terminal window and navigate back to the root of this repository. You can then send predictions to your api using the `test.py` file by running:

```bash
python test.py --local
```

Our setup has now evloved and looks like this:
![step_2](img/step_2.png)

