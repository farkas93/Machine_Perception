import tensorflow as tf
import tensorflow_io as tfio

import h5py

class GeneratorVGGNet():
  def __call__(self, filename, is_test):
    with h5py.File(filename, 'r') as hf:
      keys = list(hf.keys())
      for key in keys:
        if not is_test:
          for f, g, z in zip(hf[str(key) + "/left-eye"], hf[str(key) + "/head"], hf[str(key) + "/gaze"]) :
            yield (f, g, z)
        else: 
          for f, g in zip(hf[str(key) + "/left-eye"], hf[str(key) + "/head"]) :
            yield (f, g)


class Dataset():
  
  def __init__(self, config, path, batch_size, shuffle, is_training, is_testing):
    self.config = config

    self.is_training = is_training
    self.is_testing = is_testing
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
    
    # if (self.config['model'] == 'vggnet'):
    #   if is_training or is_testing:
    #     self.data = tf.data.Dataset.from_generator(
    #         GeneratorVGGNet(),
    #         output_types = (tf.uint8, tf.float32, tf.float32),
    #         output_shapes = (tf.TensorShape([60,90,3]), tf.TensorShape([2]), tf.TensorShape([2])),
    #         args=(self.path, False)
    #       )
    #   else:
    #     self.data = tf.data.Dataset.from_generator(
    #         GeneratorVGGNet(),
    #         output_types = (tf.uint8, tf.float32),
    #         output_shapes = (tf.TensorShape([60,90,3]), tf.TensorShape([2])),
    #         args=(self.path, True)
    #       ) 

    hdf5 = h5py.File(self.path, 'r')
    keys = list(hdf5.keys())
    
    self.left_eye = tfio.IODataset.from_hdf5(self.path, '/' + str(keys[0]) + '/left-eye', spec=tf.uint8)
    self.head = tfio.IODataset.from_hdf5(self.path, '/' + str(keys[0]) + '/head', spec=tf.float64)
    if is_training or is_testing:
      self.gaze = tfio.IODataset.from_hdf5(self.path, '/' + str(keys[0]) + '/gaze', spec=tf.float64)

    # for key in keys[1:]:
    #   temp = tfio.IODataset.from_hdf5(self.path, '/' + str(key) + '/left-eye', spec=tf.uint8)
    #   self.left_eye = self.left_eye.concatenate(temp)
    #   temp = tfio.IODataset.from_hdf5(self.path, '/' + str(key) + '/head', spec=tf.float64)
    #   self.head = self.head.concatenate(temp)
    #   if is_training or is_testing:
    #     temp = tfio.IODataset.from_hdf5(self.path, '/' + str(key) + '/gaze', spec=tf.float64)
    #     self.gaze = self.gaze.concatenate(temp)

    if is_testing or is_training:
      self.data = tf.data.Dataset.zip((self.left_eye, self.head, self.gaze))
    else:
      self.data = tf.data.Dataset.zip((self.left_eye, self.head))



    self.batch_size = batch_size
    self.shuffle = shuffle

  def get_data(self):
    """ Method used to generate and preprocess tensorflow datasets for training and test data and validation data"""
    if self.config['model'] == 'vggnet':
      if self.is_training:
        return self.data.shuffle(self.shuffle).batch(self.batch_size)
      elif self.is_testing:
        return self.data.batch(self.batch_size)
      elif not self.is_testing and not self.is_training:
        return self.data.batch(self.batch_size)
    else:
      raise NotImplementedError('In dataset.py: default input not specified for this model!')






  







     
  

