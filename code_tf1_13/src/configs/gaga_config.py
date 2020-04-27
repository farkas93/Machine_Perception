gaga_config = dict()

# General configs for training
gaga_config['n_epochs'] = 10
gaga_config['batch_size'] = 16
gaga_config['learning_rate'] = 1e-4
gaga_config['reduce_lr_after_steps'] = 15000
gaga_config['lr_multiplier_gain'] = 0.7
gaga_config['eye_patch'] = 'left-eye' #'right-eye', 'eye-region'
gaga_config['loss_terms'] = ['gaze_mse']
gaga_config['metrics'] = ['gaze_angular']

# General model configs
gaga_config['num_filters'] = [64, 128, 256, 512, 512]    # Number of filters for every convolutional layer.
gaga_config['filter_size'] = [3, 3, 3, 3, 3]             # Kernel size for convoluional layers.
gaga_config['strides'] = [1, 2, 2, 2, 2]                 # Strides for MaxPool layers
gaga_config['pool_size'] = [2, 2, 2, 2, 2]               # Pool_size for MaxPool layers
