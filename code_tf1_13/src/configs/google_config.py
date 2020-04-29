config = dict()

# General configs for training
config['n_epochs'] = 15
config['batch_size'] = 32
config['learning_rate'] = 1e-5
config['eye_patch'] = 'face' #'left-eye', 'right-eye', 'eye-region'
config['loss_terms'] = ['gaze_mse']
config['metrics'] = ['gaze_angular']

#LR stuff
config['nr_lr_reductions'] = 3
config['lr_reductions'] = [0.1, 0.1, 0.1]
config['apply_lr_reductions_at'] =  [8500, 14000, 23000, 34000]