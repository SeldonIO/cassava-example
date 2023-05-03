from helpers import preprocess
import numpy as np
import requests
from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec
import tensorflow_datasets as tfds

inference_url = 'http://localhost:8080/seldon/default/cassava/v2/models/infer'
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

res = requests.post(inference_url, json=inference_request.dict())
# res_json = json.loads(res['outputs'])


print(res)
# Parse the JSON string into a Python dictionary
response_dict = res.json()

# Extract the data array and shape from the output, assuming only one output or the target output is at index 0
data_list = response_dict["outputs"][0]["data"]
data_shape = response_dict["outputs"][0]["shape"]

# Convert the data list to a numpy array and reshape it
data_array = np.array(data_list).reshape(data_shape)

print(data_array)



