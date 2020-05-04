from typing import Dict

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation

from core import BaseDataSource, BaseModel
import util.gaze

from typing import Any, Dict, List

from configs.google_config import config
from util.pool_helper import PoolHelper
from util.lrn import LRN

import logging
logger = logging.getLogger(__name__)

class HourglassNet(BaseModel):
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

    
    def HourglassModule(inputs, order, filters, num_residual):
        """
        One Hourglass Module. Usually we stacked multiple of them together.
        https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua#L3
        inputs:
        order: The remaining order for HG modules to call itself recursively.
        num_residual: Number of residual layers for this HG module.
        """
        # Upper branch
        up1 = ResidualBlock(inputs, filters)

        for i in range(num_residual):
            up1 = ResidualBlock(up1, filters)

        # Lower branch
        low1 = MaxPool2D(pool_size=2, strides=2)(inputs)
        for i in range(num_residual):
            low1 = ResidualBlock(low1, filters)

        low2 = low1
        if order > 1:
            low2 = HourglassModule(low1, order - 1, filters, num_residual)
        else:
            for i in range(num_residual):
                low2 = ResidualBlock(low2, filters)

        low3 = low2
        for i in range(num_residual):
            low3 = ResidualBlock(low3, filters)

        up2 = tf.keras.layers.UpSampling2D(size=2)(low3)

        return up2 + up1


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)
        nr_filters = 64
        with tf.variable_scope('conv_1'):
            x = tf.keras.layers.Conv2D(
                    filters=nr_filters,
                    kernel_size=[7, 7],
                    strides = 2,
                    padding = 'same',
                    data_format='channels_first',
                    activation=None,
                    name='conv2d')(x)
            x = tf.keras.layers.BatchNormalization(axis=1)(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = ResidualBlock(x, n_filters= nr_filters, first_layer=True)
            x = ResidualBlock(x, n_filters= nr_filters)
        
        with tf.variable_scope('hourglass1'):
            x = HourglassModule(x, 4, nr_filters, 1)
        
        with tf.variable_scope('hourglass2'):
            x = HourglassModule(x, 4, nr_filters, 1)
        
        with tf.variable_scope('hourglass3'):
            x = HourglassModule(x, 4, nr_filters, 1)

        #TODO: Insert gazemap
        with tf.variable_scope('gazemap'):
        #TODO: Insert DenseNet
        with tf.variable_scope('dense_net'):
            
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