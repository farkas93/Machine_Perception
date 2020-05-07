densenet_config = dict()

# General configs for training
densenet_config['n_epochs'] = 10
densenet_config['batch_size'] = 16
densenet_config['learning_rate'] = 1e-4
densenet_config['loss_terms'] = ['gaze_mse']
densenet_config['metrics'] = ['gaze_angular']


densenet_config['growth_rate'] = 12
densenet_config['n_filters'] = 32
# densenet_config['n_dense_blocks'] = [6, 12, 24, 16] # Densenet-121
# densenet_config['n_dense_blocks'] = [6, 12, 32, 32] # Densenet-169
# densenet_config['n_dense_blocks'] = [6, 12, 48, 32] # Densenet-201
densenet_config['n_dense_blocks'] = [6, 12, 64, 48] # Densenet-264