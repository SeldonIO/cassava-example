from helpers import preprocess
import numpy as np
import requests
import json
from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec
import tensorflow_datasets as tfds

inference_url = 'http://localhost:8080/v2/models/cassava/infer'
batch_size = 9

dataset, info = tfds.load('cassava', with_info=True)

class_names = info.features['label'].names + ['unknown']

batch = dataset['validation'].map(preprocess).batch(batch_size).as_numpy_iterator()
examples = next(batch)

# Convert the TensorFlow tensor to a numpy array
input_data = np.array(examples['image'])
inference_request = InferenceRequest(
    inputs=[
        NumpyCodec.encode_input(name="payload", payload=input_data)
    ]
)

print(type(inference_request))
res = requests.post(inference_url, json=inference_request.dict())
# res_json = json.loads(res['outputs'])


print(res.json())



