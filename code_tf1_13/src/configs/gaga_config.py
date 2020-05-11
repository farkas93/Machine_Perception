gaga_config = dict()

# General configs for training
gaga_config['n_epochs'] = 20 # approx 130'000 steps
gaga_config['batch_size'] = 16
gaga_config['learning_rate'] = 1e-4
gaga_config['noise_std'] = 0.2
gaga_config['dropout_seed'] = 5


#LR stuff
gaga_config['nr_lr_reductions'] = 4
gaga_config['lr_reductions'] = [0.1, 0.01, 0.1, 0.1]
gaga_config['apply_lr_reductions_at'] =  [16600, 40250, 90000, 110000]


gaga_config['loss_terms'] = ['gaze_mse']
gaga_config['metrics'] = ['gaze_angular']

# General model configs
gaga_config['num_filters'] = [64, 128, 256, 512, 512]    # Number of filters for every convolutional layer.
gaga_config['filter_size'] = [3, 3, 3, 3, 3]             # Kernel size for convoluional layers.
gaga_config['strides'] = [1, 2, 2, 2, 2]                 # Strides for MaxPool layers
gaga_config['pool_size'] = [2, 2, 2, 2, 2]               # Pool_size for MaxPool layers