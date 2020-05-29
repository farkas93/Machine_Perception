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

from configs.gaga_config import gaga_config

class GaGaJ(BaseModel):
    """An example neural network architecture."""
    
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
        self.lr_reductions = 0


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        # self.next_step_to_reduce_lr = vgg_config['reduce_lr_after_steps']
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        left = tf.keras.backend.cast(input_tensors['left-eye'], dtype = tf.float32)
        right = tf.keras.backend.cast(input_tensors['right-eye'], dtype = tf.float32)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv_left'):
            for i, num_filters in enumerate(gaga_config['num_filters']):
                scope_name= 'conv'+str(i)
                with tf.variable_scope(scope_name):
                    if i < 2:
                        # The first two sequences between Pooling layers (only 2 convolutions) 
                        for j in range(2):
                            left = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation='relu',
                                name='conv2d')(left) 
                    else:
                        for j in range(3):
                            left = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation='relu',
                                name='conv2d')(left)
                
                # Apply pooling layer after each sequence of convolution layers
                left = tf.keras.layers.MaxPooling2D(pool_size=gaga_config['pool_size'][i], 
                                            data_format='channels_first',
                                            strides=gaga_config['strides'][i])(left)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv_right'):
            for i, num_filters in enumerate(gaga_config['num_filters']):
                scope_name= 'conv'+str(i)
                with tf.variable_scope(scope_name):
                    if i < 2:
                        # The first two sequences between Pooling layers (only 2 convolutions) 
                        for j in range(2):
                            right = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation='relu',
                                name='conv2d')(right) 
                    else:
                        for j in range(3):
                            right = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation='relu',
                                name='conv2d')(right)
                
                # Apply pooling layer after each sequence of convolution layers
                right = tf.keras.layers.MaxPooling2D(pool_size=gaga_config['pool_size'][i], 
                                            data_format='channels_first',
                                            strides=gaga_config['strides'][i])(right)

        with tf.variable_scope('fc'):
            # Create a flattened representation of the input layer
            
            left_flat = tf.keras.layers.Flatten(data_format='channels_first')(left)
            right_flat = tf.keras.layers.Flatten(data_format='channels_first')(right)

            # Concatenate head pose to our features          
            injected_layer = tf.keras.layers.concatenate([left_flat, right_flat, input_tensors['head']], axis=1)

            # FC layers      
            fc1_layer = tf.keras.layers.Dense(units=8192, activation='relu', name='fc1')(injected_layer)    

            fc1_layer = tf.keras.layers.Dropout(rate=0.5)(fc1_layer, self.is_training)          
            fc1_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc1')(fc1_layer)          
            
            fc1_layer = tf.keras.layers.Dropout(rate=0.5)(fc1_layer, self.is_training)       
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
                loss_terms[gaga_config['loss_terms'][0]] = tf.reduce_mean(tf.squared_difference(out, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics[gaga_config['metrics'][0]] = util.gaze.tensorflow_angular_error_from_pitchyaw(out, y)
        return {'gaze': out}, loss_terms, metrics

    def train_loop_post(self, current_step):
        if self.lr_reductions < gaga_config['nr_lr_reductions']:
            if current_step > gaga_config['apply_lr_reductions_at'][self.lr_reductions]:
                self._learning_rate = gaga_config['lr_reductions'][self.lr_reductions] * self._learning_rate
                self.lr_reductions += 1

    def start_training(self):
        self.train(
            num_epochs=gaga_config['n_epochs'] 
        )



class GaGaZs(BaseModel):
    """An example neural network architecture."""
    
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
        self.lr_reductions = 0


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        left = tf.keras.backend.cast(input_tensors['left-eye'], dtype = tf.float32)
        right = tf.keras.backend.cast(input_tensors['right-eye'], dtype = tf.float32)

        # Induce Noise better generalisation
        left = tf.keras.layers.GaussianNoise(stddev=gaga_config['noise_std'])(left)
        right = tf.keras.layers.GaussianNoise(stddev=gaga_config['noise_std'])(right)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv_left'):
            for i, num_filters in enumerate(gaga_config['num_filters']):
                scope_name= 'conv'+str(i)
                with tf.variable_scope(scope_name):
                    if i < 2:
                        # The first two sequences between Pooling layers (only 2 convolutions) 
                        for j in range(2):
                            left = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation=None,
                                name='conv2d')(left) 
                    else:
                        for j in range(3):
                            left = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation=None,
                                name='conv2d')(left)
                
                if i == 4:
                    left = tf.keras.layers.BatchNormalization(axis=1)(left) 
                left = tf.keras.layers.Activation('relu')(left)   
                # Apply pooling layer after each sequence of convolution layers
                left = tf.keras.layers.MaxPooling2D(pool_size=gaga_config['pool_size'][i], 
                                            data_format='channels_first',
                                            strides=gaga_config['strides'][i])(left)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv_right'):
            for i, num_filters in enumerate(gaga_config['num_filters']):
                scope_name= 'conv'+str(i)
                with tf.variable_scope(scope_name):
                    if i < 2:
                        # The first two sequences between Pooling layers (only 2 convolutions) 
                        for j in range(2):
                            right = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation=None,
                                name='conv2d')(right) 
                    else:
                        for j in range(3):
                            right = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=gaga_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation=None,
                                name='conv2d')(right)
                
                # Apply pooling layer after each sequence of convolution layers
                if i == 4:
                    right = tf.keras.layers.BatchNormalization(axis=1)(right)
                right = tf.keras.layers.Activation('relu')(right)
                right = tf.keras.layers.MaxPooling2D(pool_size=gaga_config['pool_size'][i],
                                            data_format='channels_first',
                                            strides=gaga_config['strides'][i])(right)

        with tf.variable_scope('fc'):
            # Create a flattened representation of the input layer
            
            left_flat = tf.keras.layers.Flatten(data_format='channels_first')(left)
            right_flat = tf.keras.layers.Flatten(data_format='channels_first')(right)

            left_flat = tf.keras.layers.Dropout(rate=0.25, seed=gaga_config['dropout_seed'])(left_flat, self.is_training)
            right_flat = tf.keras.layers.Dropout(rate=0.25, seed=gaga_config['dropout_seed'])(right_flat, self.is_training)
            # Concatenate head pose to our features          
            injected_layer = tf.keras.layers.concatenate([left_flat, right_flat, input_tensors['head']], axis=1)

            # FC layers           
            fc1_layer = tf.keras.layers.Dense(units=8192, activation='relu', name='fc1')(injected_layer)      
            fc1_layer = tf.keras.layers.Dropout(rate=0.6, seed=gaga_config['dropout_seed'])(fc1_layer, self.is_training)       

            fc2_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc2')(fc1_layer)
            #fc3_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc3')(fc2_layer)
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
                loss_terms[gaga_config['loss_terms'][0]] = tf.reduce_mean(tf.squared_difference(out, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics[gaga_config['metrics'][0]] = util.gaze.tensorflow_angular_error_from_pitchyaw(out, y)
        return {'gaze': out}, loss_terms, metrics

    def train_loop_post(self, current_step):
        if self.lr_reductions < gaga_config['nr_lr_reductions']:
            if current_step > gaga_config['apply_lr_reductions_at'][self.lr_reductions]:
                self._learning_rate = gaga_config['lr_reductions'][self.lr_reductions] * self._learning_rate
                self.lr_reductions += 1

    def start_training(self):
        self.train(
            num_epochs=gaga_config['n_epochs'] 
        )
