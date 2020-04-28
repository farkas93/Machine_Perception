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

from typing import Any, Dict, List

from configs.ffab_config import ffab_config

class FfabNet(BaseModel):
    """An implementation of Full-Face Appearance-Based Gaze Estimation."""
    
    def __init__(self,
                 tensorflow_session: tf.Session,
                 learning_schedule: List[Dict[str, Any]] = [],
                 train_data: Dict[str, BaseDataSource] = {},
                 test_data: Dict[str, BaseDataSource] = {},
                 test_losses_or_metrics: str = None,
                 use_batch_statistics_at_test: bool = True,
                 identifier: str = None):
        super().__init__(
            tensorflow_session = tensorflow_session, 
            learning_schedule = learning_schedule,
            train_data = train_data, 
            test_data = test_data, 
            test_losses_or_metrics = test_losses_or_metrics, 
            use_batch_statistics_at_test = use_batch_statistics_at_test, 
            identifier = identifier
            )
        self.next_step_to_reduce_lr = ffab_config['reduce_lr_after_steps']


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        # self.next_step_to_reduce_lr = vgg_config['reduce_lr_after_steps']
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv_alexnet'):
            for i, num_filters in enumerate(ffab_config['num_filters']):
                scope_name= 'conv'+str(i)
                with tf.variable_scope(scope_name):
                    x = tf.keras.layers.Conv2D(
                        filters=num_filters,
                        kernel_size=ffab_config['filter_size'][i],
                        strides=ffab_config['conv_strides'][i],
                        padding = 'same',
                        data_format='channels_first',
                        activation='relu',
                        name='conv2d')(x) 
  
                # Apply pooling layer after each sequence of convolution layers
                if i < 2:
                    x = tf.keras.layers.MaxPooling2D(pool_size=ffab_config['pool_size'][i], 
                                                data_format='channels_first',
                                                strides=ffab_config['strides'][i])(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=ffab_config['pool_size'][2], 
                                                data_format='channels_first',
                                                strides=ffab_config['strides'][2])(x)
        
        with tf.variable_scope('spatial_weights'):
            w_map = tf.keras.layers.Conv2D(
                        filters=256,
                        kernel_size= [1, 1],
                        strides=1,
                        kernel_initializer= tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                        bias_initializer = keras.initializers.Constant(0.1),
                        padding = 'same',
                        data_format='channels_first',
                        activation='relu',
                        name='conv2d')(x) 

            w_map = tf.keras.layers.Conv2D(
                        filters=256,
                        kernel_size= [1, 1],
                        strides=1,
                        kernel_initializer= tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                        bias_initializer = keras.initializers.Constant(0.1),
                        padding = 'same',
                        data_format='channels_first',
                        activation='relu',
                        name='conv2d')(w_map)

            w_map = tf.keras.layers.Conv2D(
                        filters=1,
                        kernel_size= [1, 1],
                        strides=1,
                        kernel_initializer= tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None),
                        bias_initializer = keras.initializers.Constant(1),
                        padding = 'same',
                        data_format='channels_first',
                        activation='relu',
                        name='conv2d')(w_map)

            w_map = tf.keras.backend.repeat_elements(w_map, rep = 256, axis = 1)   
            x = tf.math.multiply(x, w_map)      


        with tf.variable_scope('fc'):
            # Create a flattened representation of the input layer
            
            x_flat = tf.keras.layers.Flatten(data_format='channels_first')(x)

            # FC layers      
            fc1_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc1')(x_flat)    
            
            fc2_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc2')(fc1_layer)
            self.summary.histogram('fc2/activations', fc2_layer)

            # Directly regress two polar angles for gaze direction            
            out = tf.keras.layers.Dense(units=2, activation=None, name='output_layer')(fc2_layer)
            self.summary.histogram('output_layer/activations', out)

        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            with tf.variable_scope('mse'):  # To optimize
                # NOTE: You are allowed to change the optimized loss
                loss_terms[ffab_config['loss_terms'][0]] = tf.reduce_mean(tf.squared_difference(out, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics[ffab_config['metrics'][0]] = util.gaze.tensorflow_angular_error_from_pitchyaw(out, y)
        return {'gaze': out}, loss_terms, metrics

    def train_loop_post(self, current_step):
        if current_step > self.next_step_to_reduce_lr:
            self._learning_rate = ffab_config['lr_multiplier_gain'] * self._learning_rate
            self.next_step_to_reduce_lr += ffab_config['reduce_lr_after_steps']

    def start_training(self):
        self.train(
            num_epochs=ffab_config['n_epochs'] 
        )

