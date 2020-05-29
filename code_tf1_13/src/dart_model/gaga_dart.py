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
from operations import *



class GaGaDart(BaseModel):
    """An example neural network architecture."""
    
    def __init__(self,
                 tensorflow_session: tf.Session,
                 learning_schedule: List[Dict[str, Any]] = [],
                 train_data: Dict[str, BaseDataSource] = {},
                 test_data: Dict[str, BaseDataSource] = {},
                 test_losses_or_metrics: str = None,
                 use_batch_statistics_at_test: bool = True,
                 identifier: str = None,
                 stem_multiplier=3,
                 layer_num = 5,
                 first_C,
                 genotype):
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
        self.first_C = first_C
        self.genotype = genotype
        self.stem_multiplier = stem_multiplier
        self.layer_num = layer_num


    def Cell(self, s0, s1, genotype, C_out, reduction, reduction_prev):
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        cells_num=len(op_names) // 2
        multiplier = len(concat)

        if reduction_prev:
            s0 = FactorizedReduce(s0,C_out)
        else:
            s0 = ReLUConvBN(s0,C_out)
        s1=ReLUConvBN(s1,C_out)

        state=[s0,s1]
        offset=0
        for i in range(cells_num):
            temp=[]
            for j in range(2):
                stride = [2,2] if reduction and indices[2*i+j] < 2 else [1,1]
                h = state[indices[2*i+j]]
                temp.append(OPS[op_names[2*i+j]](h, C_out, stride))   
                #did not impelement path drop
            state.append(tf.add_n(temp))
        out=tf.concat(state[-multiplier:],axis=-1)
        return out

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
        with tf.variable_scope('conv_left',reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d,slim.separable_conv2d],activation_fn=None,padding='SAME',biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001)):
                with slim.arg_scope([slim.batch_norm],is_training=is_training):
                    C_curr = self.stem_multiplier*self.first_C
                    left_s0 = slim.conv2d(left,C_curr,[3,3],activation_fn=tflearn.relu)
                    left_s0 = slim.batch_norm(left_s0)
                    left_s1 =slim.conv2d(left,C_curr,[3,3],activation_fn=tflearn.relu)
                    left_s1=slim.batch_norm(left_s1)
                    reduction_prev = False
                    for i in range(self.layer_num):
                        if i in [self.layer_num//3, 2* self.layer_num//3]:
                            C_curr *= 2
                            reduction = True
                        else:
                            reduction = False
                        left_s0,left_s1 =left_s1,Cell(left_s0,left_s1, self.genotype, C_curr, reduction, reduction_prev)
                        reduction_prev = reduction

                    left=left_s1

        # Here, the `tf.variable_scope` scope is used to structure the
        # visualization in the Graphs tab on Tensorboard
        with tf.variable_scope('conv_right'):
            with slim.arg_scope([slim.conv2d,slim.separable_conv2d],activation_fn=None,padding='SAME',biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001)):
                with slim.arg_scope([slim.batch_norm],is_training=is_training):
                    C_curr = self.stem_multiplier*self.first_C
                    right_s0 = slim.conv2d(right,C_curr,[3,3],activation_fn=tflearn.relu)
                    right_s0 = slim.batch_norm(right_s0)
                    right_s1 = slim.conv2d(right,C_curr,[3,3],activation_fn=tflearn.relu)
                    right_s1 = slim.batch_norm(right_s1)
                    reduction_prev = False
                    for i in range(self.layer_num):
                        if i in [self.layer_num//3, 2* self.layer_num//3]:
                            C_curr *= 2
                            reduction = True
                        else:
                            reduction = False
                        right_s0,right_s1 =right_s1,Cell(right_s0,right_s1, self.genotype, C_curr, reduction, reduction_prev)
                        reduction_prev = reduction

                    right=right_s1

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
