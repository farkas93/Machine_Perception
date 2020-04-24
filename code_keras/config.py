
import time
import os

config = dict()

# Data directories
config['path_dir'] = os.getcwd()

config['data_dir'] = '../datasets'
config['log_dir'] = '../runs'

config['train_data'] = 'mp20_train.h5'
config['test_data'] = 'mp20_test_students.h5'
config['val_data'] = 'mp20_validation.h5'


# General configs for training
config['num_epochs'] = 15
config['batch_size'] = 32
config['training_steps_per_epoch'] = 1
config['validation_steps_per_epoch'] = 1
config['learning_rate'] = 1e-4

config['metrics'] = 'mse'

# Name of currently used model
config['model'] ='vggnet'

# General model configs
config['vggnet'] = {}
config['vggnet']['num_filters'] = [64, 128, 256, 512, 512]    # Number of filters for every convolutional layer.
config['vggnet']['filter_size'] = [3, 3, 3, 3, 3]             # Kernel size for convoluional layers.
config['vggnet']['strides'] = [1, 1, 2, 2, 2]                 # Strides for MaxPool layers
config['vggnet']['pool_size'] = [2, 2, 2, 2, 2]               # Pool_size for MaxPool layers
config['vggnet']['learning_rate'] = config['learning_rate']
config['vggnet']['metrics'] = config['metrics']
