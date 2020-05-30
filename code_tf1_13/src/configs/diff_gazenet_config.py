config = dict()

# General configs for training
config['n_epochs'] = 15
config['batch_size'] = 16
config['learning_rate'] = 1e-5
config['reduce_lr_after_steps'] = 20000
config['lr_multiplier_gain'] = 0.1
config['dropout_seed'] = 5
config['eye_patch'] = 'left-eye' #'right-eye', 'eye-region'
config['loss_terms'] = ['gaze_mse']
config['metrics'] = ['gaze_angular']

config['n_ref_images'] = 10

# General model configs
config['num_filters'] = [32, 32, 64]    # Number of filters for every convolutional layer.
config['filter_size'] = [5, 5, 4]             # Kernel size for convoluional layers.
config['pool_stride'] = 2                # Strides for MaxPool layers
config['pool_size'] = 2             # Pool_size for MaxPool layers
