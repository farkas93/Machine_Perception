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

from core import BaseDataSource, BaseModel
import util.gaze


class ExampleNet(BaseModel):
    """An example neural network architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['left-eye']

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv'):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(x, filters=128, kernel_size=7, strides=2,
                                     padding='same', data_format='channels_first')
                self.summary.filters('filters', x)
                self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('bn1'):
                x = tf.contrib.layers.batch_norm(x, scale=True, center=True,
                                                 is_training=self.use_batch_statistics,
                                                 trainable=True, data_format='NCHW',
                                                 updates_collections=None,
                                                 )

            with tf.variable_scope('relu1'):
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_first')
                self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('bn2'):
                x = tf.contrib.layers.batch_norm(x, scale=True, center=True,
                                                 is_training=self.use_batch_statistics,
                                                 trainable=True, data_format='NCHW',
                                                 updates_collections=None,
                                                 )

            with tf.variable_scope('relu2'):
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, filters=512, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_first')
                self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('bn3'):
                x = tf.contrib.layers.batch_norm(x, scale=True, center=True,
                                                 is_training=self.use_batch_statistics,
                                                 trainable=True, data_format='NCHW',
                                                 updates_collections=None,
                                                 )

            with tf.variable_scope('relu3'):
                x = tf.nn.relu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, filters=1024, kernel_size=5, strides=2,
                                     padding='same', data_format='channels_first')
                self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('bn4'):
                x = tf.contrib.layers.batch_norm(x, scale=True, center=True,
                                                 is_training=self.use_batch_statistics,
                                                 trainable=True, data_format='NCHW',
                                                 updates_collections=None,
                                                 )

            with tf.variable_scope('relu4'):
                x = tf.nn.relu(x)

        with tf.variable_scope('fc'):
            # Flatten the 1024 feature maps down to one vector
            x = tf.contrib.layers.flatten(x)

            # NOTE: When applying a dropout layer,
            #       do NOT forget to use training=self.is_training
            x = tf.layers.dropout(x, rate=0.5, training=self.is_training, name='drop')

            # Concatenate head pose to our features
            x = tf.concat([x, input_tensors['head']], axis=1)

            # FC layer
            x = tf.layers.dense(x, units=512, name='fc5')
            x = tf.layers.dense(x, units=256, name='fc6')
            x = tf.layers.dense(x, units=128, name='fc7')
            x = tf.layers.dense(x, units=64, name='fc8')
            self.summary.histogram('fc8/activations', x)

            # Directly regress two polar angles for gaze direction
            x = tf.layers.dense(x, units=2, name='fc9')
            self.summary.histogram('fc9/activations', x)

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
