config = dict()

# General configs for training
config['n_epochs'] = 15
config['batch_size'] = 16
config['learning_rate'] = 1e-5
config['reduce_lr_after_steps'] = 20000
config['lr_multiplier_gain'] = 0.1
config['eye_patch'] = 'face' #'right-eye', 'eye-region'
config['loss_terms'] = ['gaze_mse']
config['metrics'] = ['gaze_angular']

