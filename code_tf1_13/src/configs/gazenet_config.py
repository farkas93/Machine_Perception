gazenet_config = dict()

# General configs for training
gazenet_config['n_epochs'] = 15
gazenet_config['batch_size'] = 16
gazenet_config['learning_rate'] = 1e-5
gazenet_config['reduce_lr_after_steps'] = 20000
gazenet_config['lr_multiplier_gain'] = 0.1
gazenet_config['eye_patch'] = 'left-eye' #'right-eye', 'eye-region'
gazenet_config['loss_terms'] = ['gaze_mse']
gazenet_config['metrics'] = ['gaze_angular']

# General model configs
gazenet_config['num_filters'] = [64, 128, 256, 512, 512]    # Number of filters for every convolutional layer.
gazenet_config['filter_size'] = [3, 3, 3, 3, 3]             # Kernel size for convoluional layers.
gazenet_config['strides'] = [1, 1, 2, 2, 2]                 # Strides for MaxPool layers
gazenet_config['pool_size'] = [2, 2, 2, 2, 2]               # Pool_size for MaxPool layers
