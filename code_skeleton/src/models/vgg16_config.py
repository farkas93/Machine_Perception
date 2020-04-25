
import time
import os

vgg_config = dict()

# General configs for training
vgg_config['n_epochs'] = 20
vgg_config['batch_size'] = 32
vgg_config['learning_rate'] = 1e-5
vgg_config['eye_patch'] = 'left-eye' #'right-eye', 'eye-region'

# vgg_config['training_steps_per_epoch'] = 1
# vgg_config['validation_steps_per_epoch'] = 1
# vgg_config['metrics'] = 'mse'

# General model configs
vgg_config['num_filters'] = [64, 128, 256, 512, 512]    # Number of filters for every convolutional layer.
vgg_config['filter_size'] = [3, 3, 3, 3, 3]             # Kernel size for convoluional layers.
vgg_config['strides'] = [1, 1, 2, 2, 2]                 # Strides for MaxPool layers
vgg_config['pool_size'] = [2, 2, 2, 2, 2]               # Pool_size for MaxPool layers

