import tensorflow as tf

from tensorflow import keras 
from tensorflow.keras import Model

class VGGNet(Model):
  def __init__(self, config):
    super(VGGNet, self).__init__()
    self.config = config

  def call(self, input_layer, input_head):
    for i, num_filters in enumerate(self.config['num_filters']):
      if i < 2:
        # The first two sequences between Pooling layers (only 2 convolutions) 
        for j in range(2):
          input_layer = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=self.config['filter_size'][i],
            padding = 'same',
            activation='relu')(input_layer) 
      else:
        for j in range(3):
          input_layer = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=self.config['filter_size'][i],
            padding = 'same',
            activation='relu')(input_layer)
      # Apply pooling layer after each sequence of convolution layers
      input_layer = keras.layers.MaxPooling2D(pool_size=self.config['pool_size'][i], strides=self.config['strides'][i])(input_layer)

    # Create a flattened representation of the input layer
    flatten_layer = keras.layers.Flatten()(input_layer)

    # TODO And here we should add the head positions to the network, smh ... maybe concatenate
    injected_layer = keras.layers.concatenate([flatten_layer, input_head])

    fc1_layer = keras.Dense(units=4096, activation='relu')(injected_layer)
    fc2_layer = keras.Dense(units=4096, activation='relu')(fc1_layer)

    out = keras.Dense(units=2, activation='softmax')(fc2_layer)

    # Output is a 2D vector
    return out