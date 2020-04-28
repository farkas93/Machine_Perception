gaga_config = dict()

# General configs for training
gaga_config['n_epochs'] = 20
gaga_config['batch_size'] = 16
gaga_config['learning_rate'] = 1e-5
gaga_config['reduce_lr_after_steps'] = 40000
gaga_config['lr_multiplier_gain'] = 0.1
gaga_config['loss_terms'] = ['gaze_mse']
gaga_config['metrics'] = ['gaze_angular']

# General model configs
gaga_config['num_filters'] = [64, 128, 256, 512, 512]    # Number of filters for every convolutional layer.
gaga_config['filter_size'] = [3, 3, 3, 3, 3]             # Kernel size for convoluional layers.
gaga_config['strides'] = [1, 2, 2, 2, 2]                 # Strides for MaxPool layers
gaga_config['pool_size'] = [2, 2, 2, 2, 2]               # Pool_size for MaxPool layers
gaga_config['dropout_seed'] = 2251
