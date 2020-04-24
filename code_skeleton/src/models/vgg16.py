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
from tensorflow.keras import Model

from core import BaseDataSource, BaseModel
import util.gaze

from models.vgg16_config import config

class VGG16(BaseModel):
    """An example neural network architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        print(input_tensors)
        x = tf.keras.backend.cast(input_tensors['left-eye'], dtype = tf.float32)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv'):
            for i, num_filters in enumerate(config['num_filters']):
                scope_name= 'conv'+str(i)
                with tf.variable_scope(scope_name):
                    if i < 2:
                            # The first two sequences between Pooling layers (only 2 convolutions) 
                            for j in range(2):
                                x = tf.keras.layers.Conv2D(
                                    filters=num_filters,
                                    kernel_size=config['filter_size'][i],
                                    padding = 'same',
                                    data_format='channels_first',
                                    activation='relu',
                                    name='conv2d')(x) 
                    else:
                        for j in range(3):
                            x = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=config['filter_size'][i],
                                padding = 'same',
                                activation='relu',
                                name='conv2d')(x)

                # scope_name= 'maxpool'+str(i)
                # with tf.variable_scope(scope_name):
                # Apply pooling layer after each sequence of convolution layers
                x = tf.keras.layers.MaxPooling2D(pool_size=config['pool_size'][i], 
                                            strides=config['strides'][i])(x)

        with tf.variable_scope('fc'):
            # Create a flattened representation of the input layer
            
            x = tf.keras.layers.Flatten()(x)

            # NOTE: When applying a dropout layer,
            #       do NOT forget to use training=self.is_training

            # Concatenate head pose to our features          
            injected_layer = tf.keras.layers.concatenate([x, input_tensors['head']], axis=1)

            # FC layers           
            fc1_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc1')(injected_layer)            
            fc2_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc2')(fc1_layer)
            self.summary.histogram('fc2/activations', fc2_layer)

            # Directly regress two polar angles for gaze direction            
            out = tf.keras.layers.Dense(units=2, activation='softmax', name='output_layer')(fc2_layer)
            self.summary.histogram('output_layer/activations', out)

        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            with tf.variable_scope('mse'):  # To optimize
                # NOTE: You are allowed to change the optimized loss
                loss_terms['gaze_mse'] = tf.reduce_mean(tf.squared_difference(out, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics['gaze_angular'] = util.gaze.tensorflow_angular_error_from_pitchyaw(out, y)
        return {'gaze': x}, loss_terms, metrics

    def start_training(self):
        self.train(
            num_epochs=config['n_epochs'] 
        )
