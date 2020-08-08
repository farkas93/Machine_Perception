import tensorflow as tf

from tensorflow import keras 
from tensorflow.keras import Model

class DiffNet(Model):
  def __init__(self, config):
    super(DiffNet, self).__init__()
    self.config = config

    self.relu = keras.layers.Activation('relu')
    self.bn = keras.layers.BatchNormalization()
    self.max_pool = keras.layers.MaxPooling2D(
      pool_size=self.config['pool_size'][i], strides=self.config['strides'][i]
    )

    self.conv1 = keras.layers.Conv2D(
        filters=32,
        kernel_size=[5, 5],
        padding = 'same',
        activation=None
      ) 

    self.conv2 = keras.layers.Conv2D(
        filters=32,
        kernel_size=[5, 5],
        padding = 'same',
        activation=None
      )

    self.conv3 = keras.layers.Conv2D(
        filters=64,
        kernel_size=[4, 4],
        padding = 'same',
        activation=None
      )

    self.dense1 = keras.layers.Dense(units=256, activation='relu')
    self.dense2 = keras.layers.Dense(units=2, activation=None)

  def call(self, inputs, training = False):
    
    input_img = inputs[0]
    input_ref = inputs[1]

    # Process img:
    # First Layer - conv with 5x5 kernel, bn, relu, max_pool
    input_img = self.conv1(input_img)
    input_img = self.bn(input_img)
    input_img = self.relu(input_img)
    input_img = self.max_pool(input_img)

    # Second Layer - conv with 5x5 kernel, bn, relu, max_pool
    input_img = self.conv2(input_img)
    input_img = self.bn(input_img)
    input_img = self.relu(input_img)
    input_img = self.max_pool(input_img)

    # Third Layer - conv with 4x4 kernel, bn, relu
    input_img = self.conv3(input_img)
    input_img = self.bn(input_img)
    input_img = self.relu(input_img)

    # Process ref:
    # First Layer - conv with 5x5 kernel, bn, relu, max_pool
    input_ref = self.conv1(input_ref)
    input_ref = self.bn(input_ref)
    input_ref = self.relu(input_ref)
    input_ref = self.max_pool(input_ref)

    # Second Layer - conv with 5x5 kernel, bn, relu, max_pool
    input_ref = self.conv2(input_ref)
    input_ref = self.bn(input_ref)
    input_ref = self.relu(input_ref)
    input_ref = self.max_pool(input_ref)

    # Third Layer - conv with 4x4 kernel, bn, relu
    input_ref = self.conv3(input_ref)
    input_ref = self.bn(input_ref)
    input_ref = self.relu(input_ref)

    # Combine input and ref
    input_img = tf.keras.layers.Flatten()(input_img)
    input_ref = tf.keras.layers.Flatten()(input_ref)

    input_concat = tf.keras.layers.concatenate([input_img, input_ref])

    input_concat = self.relu(input_concat)
    input_concat = keras.layers.Dropout(rate=0.5, seed=455)(input_concat, training)

    out = self.dense1(input_concat)
    out = self.dense2(out)

    # Output is a 2D vector
    return out