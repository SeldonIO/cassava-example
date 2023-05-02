from helpers import plot, preprocess

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

tf.config.experimental.set_visible_devices([], 'GPU')

model_path = './model'
classifier = hub.KerasLayer(model_path)
batch_size = 9

dataset, info = tfds.load('cassava', with_info=True)

class_names = info.features['label'].names + ['unknown']

batch = dataset['validation'].map(preprocess).batch(batch_size).as_numpy_iterator()
examples = next(batch)
plot(examples, class_names)

predictions = classifier(examples['image'])
predictions_max = tf.argmax(predictions, axis=-1)
print(predictions_max)

plot(examples, class_names, predictions_max)
