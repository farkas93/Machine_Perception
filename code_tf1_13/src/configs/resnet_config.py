resnet_config = dict()

# General configs for training
resnet_config['n_epochs'] = 10
resnet_config['batch_size'] = 32
resnet_config['learning_rate'] = 1e-4
resnet_config['loss_terms'] = ['gaze_mse']
resnet_config['metrics'] = ['gaze_angular']

