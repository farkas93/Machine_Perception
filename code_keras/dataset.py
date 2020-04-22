import tensorflow as tf
import tensorflow_io as tfio


class Dataset():

  def __init__(self, config, path, batch_size, shuffle, is_training):
    self.config = config

    self.is_training = is_training
    self.path = path

    """
      each archive contains:
        face - a list 224x224 BGR images of type uint8
        eye-region - a list 224x60 BGR images of type uint8
        left-eye - a list 90x60 BGR images of type uint8
        right-eye - a list 90x60 BGR images of type uint8
        head - a list of 1x2 arrays. Each row contains the Euler angle representations of head orientation given in radians.
        face-landmarks - a list of 33x2 arrays. Each row contains the (u,v) coordinates of selected facial landmarks as found in the provided face image patches.
        gaze (except in test set) - a list of 1x2 arrays. Each row contains the Euler angle representations of gaze direction given in radians.
    """
    self.face = tfio.IODataset.from_hdf5(self.path, dataset='/face')
    self.eye_region = tfio.IODataset.from_hdf5(self.path, dataset='/eye-region')
    self.left_eye = tfio.IODataset.from_hdf5(self.path, dataset='/left-eye')
    self.right_eye = tfio.IODataset.from_hdf5(self.path, dataset='/right-eye')
    self.head = tfio.IODataset.from_hdf5(self.path, dataset='/head')
    self.face_landmarks = tfio.IODataset.from_hdf5(self.path, dataset='/face-landmarks')

    if self.is_training:
      self.gaze = tfio.IODataset.from_hdf5(self.path, dataset='/gaze')

    self.batch_size = batch_size
    self.shuffle = shuffle

  def get_data(self):
    """ Method used to generate tensorflow datasets for training and test data"""
    if self.config['model'] == 'vggnet':
      data = tf.data.Dataset.zip((self.left_eye, self.head, self.gaze)).shuffle(self.shuffle).batch(self.batch_size)
      return data
    else:
      raise NotImplementedError('In dataset.py: default input not specified for this model!')

  
  def get_test_data(self):
    """ Method used to generate tensorflow datasets for data with no labels/outputs provided """
    if self.config['model'] == 'vggnet':
      data = tf.data.Dataset.zip((self.left_eye, self.head)).batch(self.batch_size)
      return data
    else:
      raise NotImplementedError('In dataset.py: default input not specified for this model!')





  







     
  

