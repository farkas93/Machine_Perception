# Run this to start training, predictions and creating submission files
import os
from trainer import Trainer
from model import VGGNet
from dataset import Dataset


def main(config):

  # Read in data, prepare data to look like this (eye_patch, head_angles, true_gaze)
  # Then shuffle input data and create batches
  
  data_path = os.path.abspath(os.path.join(config['path_dir'], config['data_dir']))

  print("Loading datasets from {}\n".format(data_path))

  train_ds = Dataset(
    config,
    os.path.abspath(os.path.join(data_path, config['train_data'])),
    config['batch_size'],
    20,
    True,
    False
    )

  test_ds = Dataset(
    config,
    os.path.abspath(os.path.join(data_path, config['val_data'])),
    config['batch_size'],
    0,
    False,
    True
    )

  val_ds = Dataset(
    config,
    os.path.abspath(os.path.join(data_path, config['test_data'])),
    config['batch_size'],
    0,
    False,
    False
    )

  # Initialize model
  model = VGGNet(config['vggnet'])

  # Initalize trainer
  trainer = Trainer(config, model)

  # Train the network:
  trainer.train(train_ds.get_data(), test_ds.get_data())

  # Create preditions with trained network:
  predictions = trainer.predict(val_ds.get_data())

  # TODO Save predictions in CSV file:

  # TODO Save the current model and weights to reuse later


if __name__ == '__main__':
  from config import config
  main(config)