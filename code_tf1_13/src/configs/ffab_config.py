ffab_config = dict()

# General configs for training
ffab_config['n_epochs'] = 15
ffab_config['batch_size'] = 16
ffab_config['learning_rate'] = 1e-4
ffab_config['reduce_lr_after_steps'] = 20000
ffab_config['lr_multiplier_gain'] = 1
ffab_config['eye_patch'] = 'face'
ffab_config['loss_terms'] = ['gaze_mse']
ffab_config['metrics'] = ['gaze_angular']

# General model configs
ffab_config['num_filters'] = [96, 256, 384, 384, 256]    # Number of filters for every convolutional layer.
ffab_config['filter_size'] = [9, 5, 3, 3, 3]             # Kernel size for convoluional layers.
ffab_config['conv_strides'] = [2, 1, 1, 1, 1]            # Strides for Convolutional layers
ffab_config['strides'] = [2, 2, 2]                 # Strides for MaxPool layers
ffab_config['pool_size'] = [3, 3, 3]               # Pool_size for MaxPool layers

