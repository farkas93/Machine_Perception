
import time
import os

config = dict()

# General configs for training
config['n_epochs'] = 20
config['batch_size'] = 1024
config['training_steps_per_epoch'] = 1
config['validation_steps_per_epoch'] = 1

config['metrics'] = 'mse'

# General model configs
config['num_filters'] = [64, 128, 256, 512, 512]    # Number of filters for every convolutional layer.
config['filter_size'] = [3, 3, 3, 3, 3]             # Kernel size for convoluional layers.
config['strides'] = [1, 2, 2, 2, 2]                 # Strides for MaxPool layers
config['pool_size'] = [2, 2, 2, 2, 2]               # Pool_size for MaxPool layers
config['learning_rate'] = 1e-5
