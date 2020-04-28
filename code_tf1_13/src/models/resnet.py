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

from configs.resnet_config import resnet_config as config

class ResNet(BaseModel):
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


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""

        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)

        with tf.variable_scope('ResNet'):
            # Here, the `tf.variable_scope` scope is used to structure the
            # visualization in the Graphs tab on Tensorboard
            with tf.variable_scope('conv_1'):
                x = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[7, 7],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(x)

                x = tf.keras.layers.MaxPooling2D(
                    pool_size=[3, 3], 
                    data_format='channels_first',
                    strides=2)(x)
        
            with tf.variable_scope('conv_2'):
                for _ in range(3):
                    temp = x

                    temp = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)

                    temp = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

                    x = tf.keras.layers.add([x, temp])
                    x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
                
            with tf.variable_scope('conv_3'):
                temp = x
                temp = tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=[3, 3],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)
                
                temp = tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

                x = tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=[1, 1],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(x)
                
                x = tf.keras.layers.add([x, temp])
                x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)

                for _ in range(3):
                    temp = x

                    temp = tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)

                    temp = tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

                    x = tf.keras.layers.add([x, temp])
                    x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
                
            with tf.variable_scope('conv_4'):
                temp = x
                temp = tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=[3, 3],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)
                
                temp = tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

                x = tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=[1, 1],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(x)
                
                x = tf.keras.layers.add([x, temp])
                x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)

                for _ in range(5):
                    temp = x

                    temp = tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)

                    temp = tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

                    x = tf.keras.layers.add([x, temp])
                    x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)

             
            with tf.variable_scope('conv_5'):
                temp = x
                temp = tf.keras.layers.Conv2D(
                    filters=512,
                    kernel_size=[3, 3],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)
                
                temp = tf.keras.layers.Conv2D(
                    filters=512,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

                x = tf.keras.layers.Conv2D(
                    filters=512,
                    kernel_size=[1, 1],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(x)
                
                x = tf.keras.layers.add([x, temp])
                x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)

                for _ in range(2):
                    temp = x

                    temp = tf.keras.layers.Conv2D(
                    filters=512,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)

                    temp = tf.keras.layers.Conv2D(
                    filters=512,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

                    x = tf.keras.layers.add([x, temp])
                    x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)

            with tf.variable_scope('final'):
                x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
                x = tf.keras.layers.Dense(units=1000, activation=None, name='dense1')(x)

            with tf.variable_scope('output'):
                # Directly regress two polar angles for gaze direction            
                out = tf.keras.layers.Dense(units=2, activation=None, name='output_layer')(x)
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
        return {'gaze': out}, loss_terms, metrics


    def start_training(self):
        self.train(
            num_epochs=config['n_epochs'] 
        )


class ResNet50(BaseModel):
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
    
    def ResidualBlock(self, x, n_filters, first_layer=False, n_strides = 1):
        temp = x
        temp = tf.keras.layers.Conv2D(
                    filters=n_filters,
                    kernel_size=[1, 1],
                    strides = n_strides,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)
        
        temp = tf.keras.layers.Conv2D(
                    filters=n_filters,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)

        temp = tf.keras.layers.Conv2D(
                    filters= 4 * n_filters,
                    kernel_size=[1, 1],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

        if first_layer:
            x = tf.keras.layers.Conv2D(
                    filters= 4 * n_filters,
                    kernel_size=[1, 1],
                    strides = n_strides,
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(x)
                
        x = tf.keras.layers.add([x, temp])
        x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
        return x       

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)

        with tf.variable_scope('ResNet'):
            # Here, the `tf.variable_scope` scope is used to structure the
            # visualization in the Graphs tab on Tensorboard
            with tf.variable_scope('conv_1'):
                x = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[7, 7],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(x)

                x = tf.keras.layers.MaxPooling2D(
                    pool_size=[3, 3], 
                    data_format='channels_first',
                    strides=2)(x)
        
            with tf.variable_scope('conv_2'):
                x = self.ResidualBlock(x, 64, True)
                x = self.ResidualBlock(x, 64)
                x = self.ResidualBlock(x, 64)
                
            with tf.variable_scope('conv_3'):
                x = self.ResidualBlock(x, 128, True, 2)
                x = self.ResidualBlock(x, 128)
                x = self.ResidualBlock(x, 128)
                x = self.ResidualBlock(x, 128)
                
            with tf.variable_scope('conv_4'):
                x = self.ResidualBlock(x, 256, True, 2)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)

            with tf.variable_scope('conv_5'):
                x = self.ResidualBlock(x, 512, True, 2)
                x = self.ResidualBlock(x, 512)
                x = self.ResidualBlock(x, 512)                

            with tf.variable_scope('final'):
                x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
                x = tf.keras.layers.Dense(units=1000, activation=None, name='dense1')(x)

            with tf.variable_scope('output'):
                # Directly regress two polar angles for gaze direction            
                out = tf.keras.layers.Dense(units=2, activation=None, name='output_layer')(x)
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
        return {'gaze': out}, loss_terms, metrics


    def start_training(self):
        self.train(
            num_epochs=config['n_epochs'] 
        )


class ResNet101(BaseModel):
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
    
    def ResidualBlock(self, x, n_filters, first_layer=False, n_strides = 1):
        temp = x
        temp = tf.keras.layers.Conv2D(
                    filters=n_filters,
                    kernel_size=[1, 1],
                    strides = n_strides,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)
        
        temp = tf.keras.layers.Conv2D(
                    filters=n_filters,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)

        temp = tf.keras.layers.Conv2D(
                    filters= 4 * n_filters,
                    kernel_size=[1, 1],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

        if first_layer:
            x = tf.keras.layers.Conv2D(
                    filters= 4 * n_filters,
                    kernel_size=[1, 1],
                    strides = n_strides,
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(x)
                
        x = tf.keras.layers.add([x, temp])
        x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
        return x     

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)

        with tf.variable_scope('ResNet'):
            # Here, the `tf.variable_scope` scope is used to structure the
            # visualization in the Graphs tab on Tensorboard
            with tf.variable_scope('conv_1'):
                x = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[7, 7],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(x)

                x = tf.keras.layers.MaxPooling2D(
                    pool_size=[3, 3], 
                    data_format='channels_first',
                    strides=2)(x)
        
            with tf.variable_scope('conv_2'):
                x = self.ResidualBlock(x, 64, True)
                x = self.ResidualBlock(x, 64)
                x = self.ResidualBlock(x, 64)
                
            with tf.variable_scope('conv_3'):
                x = self.ResidualBlock(x, 128, True, 2)
                x = self.ResidualBlock(x, 128)
                x = self.ResidualBlock(x, 128)
                x = self.ResidualBlock(x, 128)
                
            with tf.variable_scope('conv_4'):
                x = self.ResidualBlock(x, 256, True, 2)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)

            with tf.variable_scope('conv_5'):
                x = self.ResidualBlock(x, 512, True, 2)
                x = self.ResidualBlock(x, 512)
                x = self.ResidualBlock(x, 512)                

            with tf.variable_scope('final'):
                x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
                x = tf.keras.layers.Dense(units=1000, activation=None, name='dense1')(x)

            with tf.variable_scope('output'):
                # Directly regress two polar angles for gaze direction            
                out = tf.keras.layers.Dense(units=2, activation=None, name='output_layer')(x)
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
        return {'gaze': out}, loss_terms, metrics


    def start_training(self):
        self.train(
            num_epochs=config['n_epochs'] 
        )


class ResNet152(BaseModel):
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
    
    def ResidualBlock(self, x, n_filters, first_layer=False, n_strides = 1):
        temp = x
        temp = tf.keras.layers.Conv2D(
                    filters=n_filters,
                    kernel_size=[1, 1],
                    strides = n_strides,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)
        
        temp = tf.keras.layers.Conv2D(
                    filters=n_filters,
                    kernel_size=[3, 3],
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(temp)

        temp = tf.keras.layers.Conv2D(
                    filters= 4 * n_filters,
                    kernel_size=[1, 1],
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(temp)

        if first_layer:
            x = tf.keras.layers.Conv2D(
                    filters= 4 * n_filters,
                    kernel_size=[1, 1],
                    strides = n_strides,
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(x)
                
        x = tf.keras.layers.add([x, temp])
        x = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
        return x    

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)

        with tf.variable_scope('ResNet'):
            # Here, the `tf.variable_scope` scope is used to structure the
            # visualization in the Graphs tab on Tensorboard
            with tf.variable_scope('conv_1'):
                x = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[7, 7],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation='relu',
                    name='conv2d')(x)

                x = tf.keras.layers.MaxPooling2D(
                    pool_size=[3, 3], 
                    data_format='channels_first',
                    strides=2)(x)
        
            with tf.variable_scope('conv_2'):
                x = self.ResidualBlock(x, 64, True)
                x = self.ResidualBlock(x, 64)
                x = self.ResidualBlock(x, 64)
                
            with tf.variable_scope('conv_3'):
                x = self.ResidualBlock(x, 128, True, 2)
                x = self.ResidualBlock(x, 128)
                x = self.ResidualBlock(x, 128)
                x = self.ResidualBlock(x, 128)
                
            with tf.variable_scope('conv_4'):
                x = self.ResidualBlock(x, 256, True, 2)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)
                x = self.ResidualBlock(x, 256)

            with tf.variable_scope('conv_5'):
                x = self.ResidualBlock(x, 512, True, 2)
                x = self.ResidualBlock(x, 512)
                x = self.ResidualBlock(x, 512)                

            with tf.variable_scope('final'):
                x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
                x = tf.keras.layers.Dense(units=1000, activation=None, name='dense1')(x)

            with tf.variable_scope('output'):
                # Directly regress two polar angles for gaze direction            
                out = tf.keras.layers.Dense(units=2, activation=None, name='output_layer')(x)
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
        return {'gaze': out}, loss_terms, metrics


    def start_training(self):
        self.train(
            num_epochs=config['n_epochs'] 
        )