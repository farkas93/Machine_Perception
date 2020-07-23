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

class GaGooGagLeNet(BaseModel):
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
        """Build model after template from https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14"""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x_eye = tf.keras.backend.cast(input_tensors['eye-region'], dtype = tf.float32)
        x_face = tf.keras.backend.cast(input_tensors['face'], dtype = tf.float32)
              
              
        eye_loss1_classifier, eye_loss2_classifier, eye_loss3_classifier = self.google_net(x_eye, 'Eye_')
        face_loss1_classifier, face_loss2_classifier, face_loss3_classifier = self.google_net(x_face, 'Face_')

        out_eye_concat = self.output_concat_layer(eye_loss1_classifier, eye_loss2_classifier, eye_loss3_classifier, 
                                            input_tensors['head'], 'Eye_')

        flat_face_landmarks = Flatten(data_format='channels_first')(input_tensors['face-landmarks'])                                    
        out_face_concat = self.output_concat_layer(face_loss1_classifier, face_loss2_classifier, face_loss3_classifier, 
                                            flat_face_landmarks, 'Face_')
        with tf.variable_scope('Final_Output'):
            
            output = Concatenate(axis=1, name='concat_final')([out_eye_concat, out_face_concat])
            output = Dense(1000, name='final_fc', kernel_regularizer=l2(0.0002))(output) 
            output = tf.keras.layers.BatchNormalization(axis=1)(output)
            output = Dropout(rate=0.2)(output)
            output = Dense(units=2, activation=None, name='gaze_fc')(output)
            self.summary.histogram('output_layer/activations', output)

        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            with tf.variable_scope('mse'):  # To optimize
                # NOTE: You are allowed to change the optimized loss
                loss_terms[config['loss_terms'][0]] = tf.reduce_mean(tf.squared_difference(output, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics[config['metrics'][0]] = util.gaze.tensorflow_angular_error_from_pitchyaw(output, y)
        return {'gaze': output}, loss_terms, metrics
   
    def train_loop_post(self, current_step):
        if self.lr_reductions < config['nr_lr_reductions']:
            if current_step > config['apply_lr_reductions_at'][self.lr_reductions]:
                self._learning_rate = config['lr_reductions'][self.lr_reductions] * self._learning_rate
                self.lr_reductions += 1

    def start_training(self):
        self.train(
            num_epochs=config['n_epochs'] 
        )

    def google_net(self, input, name_prefix):
        input_pad = ZeroPadding2D(padding=(3, 3), data_format='channels_first')(input)
        with tf.variable_scope(name_prefix + 'LRN1'):
            conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='valid', data_format='channels_first', activation='relu', name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(input_pad)
            conv1_zero_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(conv1_7x7_s2)
            pool1_helper = PoolHelper()(conv1_zero_pad) #TODO: Find out if we have to change the method cuz of channels_first        
            pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format='channels_first', name='pool1/3x3_s2')(pool1_helper)
            pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2) #TODO: Find out if we have to change the method cuz of channels_first

        with tf.variable_scope(name_prefix + 'LRN2'):
            conv2_3x3_reduce = Conv2D(64, (1,1), padding='same', data_format='channels_first', activation='relu', name='conv2/3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_norm1)
            conv2_3x3 = Conv2D(192, (3,3), padding='same', data_format='channels_first', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
            conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3) #TODO: Find out if we have to change the method cuz of channels_first
            conv2_zero_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(conv2_norm2)
            pool2_helper = PoolHelper()(conv2_zero_pad) #TODO: Find out if we have to change the method cuz of channels_first
            pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format='channels_first', name='pool2/3x3_s2')(pool2_helper)
        
        with tf.variable_scope(name_prefix + 'Inception3a'):
            inception_3a_1x1 = Conv2D(64, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3a/1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
            inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
            inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_3a_3x3_reduce)
            inception_3a_3x3 = Conv2D(128, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_3a/3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
            inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
            inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_3a_5x5_reduce)
            inception_3a_5x5 = Conv2D(32, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_3a/5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
            inception_3a_pool = MaxPooling2D(pool_size=(3,3), data_format='channels_first', strides=(1,1), padding='same', name='inception_3a/pool')(pool2_3x3_s2)
            inception_3a_pool_proj = Conv2D(32, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
            inception_3a_output = Concatenate(axis=1, name='inception_3a/output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

        with tf.variable_scope(name_prefix + 'Inception3b'):
            inception_3b_1x1 = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3b/1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)
            inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
            inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_3b_3x3_reduce)
            inception_3b_3x3 = Conv2D(192, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_3b/3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
            inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
            inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_3b_5x5_reduce)
            inception_3b_5x5 = Conv2D(96, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_3b/5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
            inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', name='inception_3b/pool')(inception_3a_output)
            inception_3b_pool_proj = Conv2D(64, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
            inception_3b_output = Concatenate(axis=1, name='inception_3b/output')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])

            inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_3b_output)
            pool3_helper = PoolHelper()(inception_3b_output_zero_pad) #TODO: Find out if we have to change the method cuz of channels_first
            pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format='channels_first', name='pool3/3x3_s2')(pool3_helper)

        with tf.variable_scope(name_prefix + 'Inception4a'):
            inception_4a_1x1 = Conv2D(192, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4a/1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
            inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
            inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_4a_3x3_reduce)
            inception_4a_3x3 = Conv2D(208, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_4a/3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
            inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
            inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_4a_5x5_reduce)
            inception_4a_5x5 = Conv2D(48, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_4a/5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
            inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same',data_format='channels_first', name='inception_4a/pool')(pool3_3x3_s2)
            inception_4a_pool_proj = Conv2D(64, (1,1), padding='same',data_format='channels_first', activation='relu', name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
            inception_4a_output = Concatenate(axis=1, name='inception_4a/output')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])
           
        with tf.variable_scope(name_prefix + 'Output1'):
            loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), data_format='channels_first', name='loss1/ave_pool')(inception_4a_output)
            loss1_conv = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='loss1/conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
            loss1_flat = Flatten(data_format='channels_first')(loss1_conv)
            loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)
            loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
            loss1_classifier = Dense(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)

        with tf.variable_scope(name_prefix + 'Inception4b'):
            inception_4b_1x1 = Conv2D(160, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4b/1x1', kernel_regularizer=l2(0.0002))(inception_4a_output)
            inception_4b_3x3_reduce = Conv2D(112, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
            inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_4b_3x3_reduce)
            inception_4b_3x3 = Conv2D(224, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_4b/3x3', kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
            inception_4b_5x5_reduce = Conv2D(24, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
            inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_4b_5x5_reduce)
            inception_4b_5x5 = Conv2D(64, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_4b/5x5', kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
            inception_4b_pool = MaxPooling2D(pool_size=(3,3), data_format='channels_first', strides=(1,1), padding='same', name='inception_4b/pool')(inception_4a_output)
            inception_4b_pool_proj = Conv2D(64, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4b/pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
            inception_4b_output = Concatenate(axis=1, name='inception_4b/output')([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj])

        with tf.variable_scope(name_prefix + 'Inception4c'):
            inception_4c_1x1 = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4c/1x1', kernel_regularizer=l2(0.0002))(inception_4b_output)
            inception_4c_3x3_reduce = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4c/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
            inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_4c_3x3_reduce)
            inception_4c_3x3 = Conv2D(256, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_4c/3x3', kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
            inception_4c_5x5_reduce = Conv2D(24, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4c/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
            inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_4c_5x5_reduce)
            inception_4c_5x5 = Conv2D(64, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_4c/5x5', kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
            inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', name='inception_4c/pool')(inception_4b_output)
            inception_4c_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', data_format='channels_first', name='inception_4c/pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
            inception_4c_output = Concatenate(axis=1, name='inception_4c/output')([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj])

        with tf.variable_scope(name_prefix + 'Inception4d'):
            inception_4d_1x1 = Conv2D(112, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4d/1x1', kernel_regularizer=l2(0.0002))(inception_4c_output)
            inception_4d_3x3_reduce = Conv2D(144, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4d/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
            inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_4d_3x3_reduce)
            inception_4d_3x3 = Conv2D(288, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_4d/3x3', kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
            inception_4d_5x5_reduce = Conv2D(32, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4d/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
            inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_4d_5x5_reduce)
            inception_4d_5x5 = Conv2D(64, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_4d/5x5', kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
            inception_4d_pool = MaxPooling2D(pool_size=(3,3), data_format='channels_first', strides=(1,1), padding='same', name='inception_4d/pool')(inception_4c_output)
            inception_4d_pool_proj = Conv2D(64, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4d/pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
            inception_4d_output = Concatenate(axis=1, name='inception_4d/output')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])

        with tf.variable_scope(name_prefix + 'Output2'):
            loss2_ave_pool = AveragePooling2D(pool_size=(5,5), data_format='channels_first', strides=(3,3), name='loss2/ave_pool')(inception_4d_output)
            loss2_conv = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='loss2/conv', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
            loss2_flat = Flatten(data_format='channels_first')(loss2_conv)
            loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)
            loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
            loss2_classifier = Dense(1000, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)
        
        
        with tf.variable_scope(name_prefix + 'Inception4e'):
            inception_4e_1x1 = Conv2D(256, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4e/1x1', kernel_regularizer=l2(0.0002))(inception_4d_output)
            inception_4e_3x3_reduce = Conv2D(160, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4e/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
            inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_4e_3x3_reduce)
            inception_4e_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', data_format='channels_first', name='inception_4e/3x3', kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
            inception_4e_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', data_format='channels_first', name='inception_4e/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
            inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_4e_5x5_reduce)
            inception_4e_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', data_format='channels_first', name='inception_4e/5x5', kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
            inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', name='inception_4e/pool')(inception_4d_output)
            inception_4e_pool_proj = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_4e/pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)
            inception_4e_output = Concatenate(axis=1, name='inception_4e/output')([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj])

            inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_4e_output)
            pool4_helper = PoolHelper()(inception_4e_output_zero_pad) #TODO: Find out if we have to change the method cuz of channels_first
            pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool4/3x3_s2')(pool4_helper)

        with tf.variable_scope(name_prefix + 'Inception5a'):
            inception_5a_1x1 = Conv2D(256, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5a/1x1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
            inception_5a_3x3_reduce = Conv2D(160, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
            inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_5a_3x3_reduce)
            inception_5a_3x3 = Conv2D(320, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_5a/3x3', kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
            inception_5a_5x5_reduce = Conv2D(32, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
            inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_5a_5x5_reduce)
            inception_5a_5x5 = Conv2D(128, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_5a/5x5', kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
            inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', name='inception_5a/pool')(pool4_3x3_s2)
            inception_5a_pool_proj = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5a/pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)
            inception_5a_output = Concatenate(axis=1, name='inception_5a/output')([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj])

        with tf.variable_scope(name_prefix + 'Inception5b'):
            inception_5b_1x1 = Conv2D(384, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5b/1x1', kernel_regularizer=l2(0.0002))(inception_5a_output)
            inception_5b_3x3_reduce = Conv2D(192, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
            inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(inception_5b_3x3_reduce)
            inception_5b_3x3 = Conv2D(384, (3,3), padding='valid', data_format='channels_first', activation='relu', name='inception_5b/3x3', kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
            inception_5b_5x5_reduce = Conv2D(48, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
            inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inception_5b_5x5_reduce)
            inception_5b_5x5 = Conv2D(128, (5,5), padding='valid', data_format='channels_first', activation='relu', name='inception_5b/5x5', kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
            inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', name='inception_5b/pool')(inception_5a_output)
            inception_5b_pool_proj = Conv2D(128, (1,1), padding='same', data_format='channels_first', activation='relu', name='inception_5b/pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)
            inception_5b_output = Concatenate(axis=1, name='inception_5b/output')([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj])
        
        with tf.variable_scope(name_prefix + 'Output3'):
            pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), data_format='channels_first', name='pool5/7x7_s2')(inception_5b_output)
            loss3_flat = Flatten(data_format='channels_first')(pool5_7x7_s1)
            pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
            loss3_classifier = Dense(1000, name='loss3/classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
        return loss1_classifier, loss2_classifier, loss3_classifier

    def output_concat_layer(self, out1, out2, out3, additional_input, name_prefix): 
        with tf.variable_scope(name_prefix+'Output_Concat'):       
            output = Concatenate(axis=1, name='concat_output1')([out1,out2])            
            output = Dense(1000, name='concat_picker1', kernel_regularizer=l2(0.0002))(output) 
            output = tf.keras.layers.BatchNormalization(axis=1)(output)
            output = Dropout(rate=0.6)(output)
            output = Concatenate(axis=1, name='concat_output2')([output,out3, additional_input])
            output = Dense(1000, name='concat_picker2', kernel_regularizer=l2(0.0002))(output) 
            output = tf.keras.layers.BatchNormalization(axis=1)(output)
            return output