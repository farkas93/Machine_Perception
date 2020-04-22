# Run this to start training, predictions and creating submission files
from trainer import Trainer
from model import VGGNet
from dataset import Dataset

def main(config):

  # Read in data, prepare data to look like this (eye_patch, head_angles, true_gaze)
  # Then shuffle input data and create batches

  train_ds = None
  test_ds = None

  val_ds = None

  # Initialize model
  model = VGGNet(config['vggnet'])

  # Initalize trainer
  trainer = Trainer(config, model)

  # Train the network:
  trainer.train(train_ds, test_ds)

  # Create preditions with trained network:
  predictions = trainer.predict(val_ds)

  # TODO Save predictions in CSV file:

  # TODO Save the current model and weights to reuse later


if __name__ == '__main__':
  from config import config
  main(config)