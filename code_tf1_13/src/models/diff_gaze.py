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

from configs.diff_gazenet_config import config as net_config

class DiffGazeNet(BaseModel):
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
        self.next_step_to_reduce_lr = net_config['reduce_lr_after_steps']

    def conv_block(self, input1, input2, kernel_size, filters, block_nr, with_maxpool):
        #Define shared layers
        conv = tf.keras.layers.Conv2D(
                                filters=filters,
                                kernel_size=kernel_size,
                                padding = 'same',
                                data_format='channels_first',
                                activation=None,
                                name='conv2d_'+block_nr)
        bn = tf.keras.layers.BatchNormalization(axis=1)
        relu = tf.keras.layers.Activation('relu')

        #run layers on input1
        out1 = conv(input1)
        out1 = bn(out1)
        out1 = relu(out1)
        
        #run layers on input2
        out2 = conv(input2)
        out2 = bn(out2)
        out2 = relu(out2)

        if with_maxpool:
            max_pool = tf.keras.layers.MaxPooling2D(pool_size=net_config['pool_size'], 
                                            data_format='channels_first',
                                            strides=net_config['pool_stride'])
            out1 = max_pool(out1)
            out2 = max_pool(out2)
        return out1, out2

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        #TODO: Rewrite input extraction according to Jans implementation
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = tf.keras.backend.cast(input_tensors[net_config['eye_patch']], dtype = tf.float32)

        #Downscale input so sizes are approximately the same as in the GazeNet paper
        x = tf.keras.layers.MaxPooling2D(pool_size=2, 
                                    data_format='channels_first',
                                    strides=2)(x)

        diff_x = tf.keras.backend.cast(input_tensors[net_config['eye_patch'] + '_ref'], dtype = tf.float32)

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv'):
            x, diff_x = self.conv_block(input1=x, input2=diff_x, kernel_size=net_config['filter_size'][0], 
                                    filters=net_config['num_filters'][0], block_nr='_0', with_maxpool=True)
            x, diff_x = self.conv_block(input1=x, input2=diff_x, kernel_size=net_config['filter_size'][1], 
                                    filters=net_config['num_filters'][1], block_nr='_1', with_maxpool=True)
            x, diff_x = self.conv_block(input1=x, input2=diff_x, kernel_size=net_config['filter_size'][2], 
                                    filters=net_config['num_filters'][2], block_nr='_2', with_maxpool=False)

        with tf.variable_scope('fc'):
            # Create a flattened representation of the input layer
            
            x = tf.keras.layers.Flatten(data_format='channels_first')(x)
            diff_x = tf.keras.layers.Flatten(data_format='channels_first')(diff_x)

            # Concatenate head pose to our features
            concat_layer = tf.keras.layers.concatenate([x, diff_x], axis=1)                       
            #injected_layer = tf.keras.layers.concatenate([x, input_tensors['head']], axis=1)
            
            fc1_layer = tf.keras.layers.Activation('relu')(concat_layer)
            fc1_layer = tf.keras.layers.Dropout(rate=0.5, seed=net_config['dropout_seed'])(fc1_layer, self.is_training) 
            # FC layers      
            fc1_layer = tf.keras.layers.Dense(units=256, activation='relu', name='fc1')(fc1_layer)
            self.summary.histogram('fc2/activations', fc1_layer)    

            # Directly regress two polar angles for gaze direction            
            out = tf.keras.layers.Dense(units=2, activation=None, name='output_layer')(fc1_layer)
            self.summary.histogram('output_layer/activations', out)

        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            y_ref = input_tensors['gaze_ref']
            with tf.variable_scope('L_diff'):  # To optimize
                # NOTE: You are allowed to change the optimized loss
                loss_terms[net_config['loss_terms'][0]] = tf.norm(out - (y - y_ref))
        return {'gaze': out}, loss_terms, metrics

    def train_loop_post(self, current_step):
        if current_step > self.next_step_to_reduce_lr:
            self._learning_rate = net_config['lr_multiplier_gain'] * self._learning_rate
            self.next_step_to_reduce_lr += net_config['reduce_lr_after_steps']

    def start_training(self):
        self.train(
            num_epochs=net_config['n_epochs'] 
        )
