"""Copyright (c) 2020 AIT Lab, ETH Zurich

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

"""Example architecture."""
from typing import Dict

import tensorflow as tf
from tensorflow import keras

from core import BaseDataSource, BaseModel
import util.gaze

from configs.densenet_config import densenet_config as config

class DenseNet(BaseModel):
    """A DenseNet implementation"""
      
    def CompositeBlock(self, x, n_filters, is_bottleneck = False):
        with tf.variable_scope('composite_block'):
          if is_bottleneck:
            x = tf.keras.layers.BatchNormalization(axis = 1)(x, self.use_batch_statistics)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(
                          filters= 4 * n_filters,
                          kernel_size=[1, 1],
                          padding = 'same',
                          data_format='channels_first',
                          activation=None,
                          name='conv2d')(x)

          x = tf.keras.layers.BatchNormalization(axis = 1)(x, self.use_batch_statistics)
          x = tf.keras.layers.ReLU()(x)
          x = tf.keras.layers.Conv2D(
                        filters=n_filters,
                        kernel_size=[3, 3],
                        padding = 'same',
                        data_format='channels_first',
                        activation=None,
                        name='conv2d')(x)
          return x

    def DenseBlock(self, x, n_layers, growth_rate, is_bottleneck = False):
        with tf.variable_scope('dense_block'):
          for _ in range(n_layers):
            temp = x
            x = self.CompositeBlock(x, growth_rate, is_bottleneck)
            x = tf.keras.layers.concatenate([temp, x], axis=1)          
          return x

    def TransitionBlock(self, x, n_filters, n_strides = [2, 2]):
        with tf.variable_scope('transition_block'):
          x = tf.keras.layers.BatchNormalization(axis = 1)(x, self.use_batch_statistics)
          x = tf.keras.layers.Conv2D(
                        filters= n_filters,
                        kernel_size=[1, 1],
                        padding = 'same',
                        data_format='channels_first',
                        activation=None,
                        name='conv2d')(x)
          x = tf.keras.layers.AveragePooling2D(pool_size = [2, 2], strides = n_strides, data_format='channels_first')(x)
          return x


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)

        growth_rate = config['growth_rate']
        n_filters = config['n_filters']
        n_dense_blocks = config['n_dense_blocks']

        with tf.variable_scope('DenseNet'):

          with tf.variable_scope('first_block'):
            x = tf.keras.layers.Conv2D(
                        filters= n_filters,
                        kernel_size=[7, 7],
                        strides = [2, 2],
                        padding = 'same',
                        data_format='channels_first',
                        activation=None,
                        name='conv2d')(x)
            x = tf.keras.layers.AveragePooling2D(pool_size = [3, 3], strides = [2, 2], data_format='channels_first')(x)
          
          for i, n_layers in enumerate(n_dense_blocks):
            with tf.variable_scope('DenseBlock'):
              x = self.DenseBlock(x = x, n_layers = n_layers, growth_rate = growth_rate, is_bottleneck = True)
              n_filters += (n_layers - 1) * growth_rate
              if i < 3:
                x = self.TransitionBlock(x, n_filters)
              else:
                x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)

          with tf.variable_scope('output'):
            x = tf.keras.layers.Dense(units=2, activation = None)(x)
        
        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            with tf.variable_scope('mse'):  # To optimize
                # NOTE: You are allowed to change the optimized loss
                loss_terms['gaze_mse'] = tf.reduce_mean(tf.squared_difference(x, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics['gaze_angular'] = util.gaze.tensorflow_angular_error_from_pitchyaw(x, y)
        return {'gaze': x}, loss_terms, metrics
    
    def start_training(self):
        self.train(
            num_epochs=config['n_epochs'] 
        )
