from helpers import preprocess
import numpy as np
import requests
import json
from mlserver.types import InferenceRequest
import tensorflow_datasets as tfds

inference_url = 'http://localhost:8080/v2/models/cassava/infer'
batch_size = 9

dataset, info = tfds.load('cassava', with_info=True)

class_names = info.features['label'].names + ['unknown']

batch = dataset['validation'].map(preprocess).batch(batch_size).as_numpy_iterator()
examples = next(batch)

# Convert the TensorFlow tensor to a numpy array
input_data = np.array(examples['image'])
input_list = input_data.tolist()
print(type(input_list))
print(input_data.shape)
print(input_data)

# Build the MLServer's inference request
request_dict = {
    "inputs": [
        {
            "name": "image",  # Replace with the actual input name for the model
            "shape": input_data.shape,
            "datatype": "FP32",  # Use "INT32" for integer arrays, etc.
            "data": input_list
        }
    ]
}

print(type(request_dict))
res = requests.post(inference_url, json=request_dict)
res_json = json.loads(res['outputs'])


print(res_json)



