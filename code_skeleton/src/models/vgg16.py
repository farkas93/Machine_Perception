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

from models.vgg16_config import vgg_config

class VGG16(BaseModel):
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
        self.next_step_to_reduce_lr = vgg_config['reduce_lr_after_steps']


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        # self.next_step_to_reduce_lr = vgg_config['reduce_lr_after_steps']
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors[vgg_config['eye_patch']], dtype = tf.float32)

        #Downscale input
        x = tf.keras.layers.MaxPooling2D(pool_size=2, 
                                    data_format='channels_first',
                                    strides=2)(x)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv'):
            for i, num_filters in enumerate(vgg_config['num_filters']):
                scope_name= 'conv'+str(i)
                with tf.variable_scope(scope_name):
                    if i < 2:
                        # The first two sequences between Pooling layers (only 2 convolutions) 
                        for j in range(2):
                            x = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=vgg_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation='relu',
                                name='conv2d')(x) 
                    else:
                        for j in range(3):
                            x = tf.keras.layers.Conv2D(
                                filters=num_filters,
                                kernel_size=vgg_config['filter_size'][i],
                                padding = 'same',
                                data_format='channels_first',
                                activation='relu',
                                name='conv2d')(x)
                
                # Apply pooling layer after each sequence of convolution layers
                x = tf.keras.layers.MaxPooling2D(pool_size=vgg_config['pool_size'][i], 
                                            data_format='channels_first',
                                            strides=vgg_config['strides'][i])(x)

        with tf.variable_scope('fc'):
            # Create a flattened representation of the input layer
            
            x = tf.keras.layers.Flatten(data_format='channels_first')(x)

            x = tf.keras.layers.Dropout(rate=0.5, data_format='channels_first', name='dropout1')(x)
            # Concatenate head pose to our features          
            #injected_layer = tf.keras.layers.concatenate([x, input_tensors['head']], axis=1)

            # FC layers           
            fc1_layer = tf.keras.layers.Dense(units=4096, activation='relu', name='fc1')(x)             
            fc1_layer = tf.keras.layers.Dropout(rate=0.5, data_format='channels_first', name='dropout2')(fc1_layer)       

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
                loss_terms[vgg_config['loss_terms'][0]] = tf.reduce_mean(tf.squared_difference(out, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics[vgg_config['metrics'][0]] = util.gaze.tensorflow_angular_error_from_pitchyaw(out, y)
        return {'gaze': out}, loss_terms, metrics

    def train_loop_post(self, current_step):
        if current_step > self.next_step_to_reduce_lr:
            self._learning_rate = vgg_config['lr_multiplier_gain'] * self._learning_rate
            self.next_step_to_reduce_lr += vgg_config['reduce_lr_after_steps']

    def start_training(self):
        self.train(
            num_epochs=vgg_config['n_epochs'] 
        )
