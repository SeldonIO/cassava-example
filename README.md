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

Then we can simply run:

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

## Containerizing The Model

[Containers](https://en.wikipedia.org/wiki/Containerization_(computing)) are an easy way to package our application together with it's runtime and dependencies. More importantly, containerizing our model allows it to run in a variety of different environments. 

> **Note:** you will need [Docker](https://www.docker.com/) installed to run this section of the tutorial. You'll also need a [docker hub](https://hub.docker.com/) account or another container registry.

Taking our model and packaging it into a container manually can be a pretty tricky process and requires knowledge of writing Dockerfiles. Thankfully MLServer removes this complexity and provides us with a simple `build` command.

Before we run this command, we need to provide our dependencies in either a `requirements.txt` or a `conda.env` file. The requirements file we'll use for this example is stored in `model/requirements.txt`:

```
tensorflow==2.12.0
tensorflow-hub==0.13.0
```

> Notice that we didn't need to include `mlserver` in our requirements? That's because the builder image has mlserver included already.

We're now ready to build our container image using:

```bash
mlserver build model/ -t [YOUR_CONTAINER_REGISTRY]/[IMAGE_NAME]
```

Make sure you replace `YOUR_CONTAINER_REGISTRY` and `IMAGE_NAME` with your dockerhub username and a suitable name e.g. "bobsmith/cassava".

MLServer will now build the model into a container image for us. We can check the output of this by running:

```bash
docker images
```

Finally, we want to send this container image to be stored in our container registry. We can do this by running:

```bash
docker push [YOUR_CONTAINER_REGISTRY]/[IMAGE_NAME]
```

Our setup now looks like this. Where our model has been packaged and sent to a container registry:
![step_3](img/step_3.png)

## Deploying to Kubernetes

Now that we've turned our model into a production-ready API, containerized it and pushed it to a registry, it's time to deploy our model.

We're going to use a popular open source framework called [Seldon Core](https://github.com/seldonio/seldon-core) to deploy our model. Seldon Core is great because it combines all of the awesome cloud-native features we get from [Kubernetes](https://kubernetes.io/) but it also adds machine-learning specific features.

*This tutorial assumes you already have a Seldon Core cluster up and running. If that's not the case, head over the [installation instructions](https://docs.seldon.io/projects/seldon-core/en/latest/nav/installation.html) and get set up first. You'll also need to install the `kubectl` command line interface.*

To create our deployment with Seldon Core we need to create a small configuration file that looks like this:

*You can find this file named `deployment.yaml` in the base folder of this tutorial's repository.*

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: cassava
spec:
  protocol: v2
  predictors:
    - componentSpecs:
        - spec:
            containers:
              - image: YOUR_CONTAINER_REGISTRY/IMAGE_NAME
                name: cassava
                imagePullPolicy: Always
      graph:
        name: cassava
        type: MODEL
      name: cassava
```

Make sure you replace `YOUR_CONTAINER_REGISTRY` and `IMAGE_NAME` with your dockerhub username and a suitable name e.g. "bobsmith/cassava".

