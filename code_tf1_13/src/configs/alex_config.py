alex_config = dict()

# General configs for training
alex_config['n_epochs'] = 15
alex_config['batch_size'] = 16
alex_config['learning_rate'] = 1e-4
alex_config['reduce_lr_after_steps'] = 20000
alex_config['lr_multiplier_gain'] = 1
alex_config['eye_patch'] = 'left-eye' #'right-eye', 'eye-region'
alex_config['loss_terms'] = ['gaze_mse']
alex_config['metrics'] = ['gaze_angular']

# General model configs
alex_config['num_filters'] = [96, 256, 384, 384, 256]    # Number of filters for every convolutional layer.
alex_config['filter_size'] = [9, 5, 3, 3, 3]             # Kernel size for convoluional layers.
alex_config['conv_strides'] = [2, 1, 1, 1, 1]            # Strides for Convolutional layers
alex_config['strides'] = [2, 2, 2]                 # Strides for MaxPool layers
alex_config['pool_size'] = [3, 3, 3]               # Pool_size for MaxPool layers

