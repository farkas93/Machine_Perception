import tensorflow as tf
from tensorflow import keras 

class Trainer:

  def __init__(self, config, model):
    self.config = config

    # Generate Model to train:
    self.model = model # VGGNet(config['vggnet'])

    # Used loss function MSE:
    self.loss_fn = keras.losses.MeanSquaredError()

    # Used optimizer ADAM
    self.optimizer_fn = keras.optimizers.Adam(lr = self.config['learning_rate'])

    # Metrics used to track loss of train and evaluation:
    self.train_loss = keras.metrics.Mean(name='train_loss')
    self.test_loss = keras.metrics.Mean(name='test_loss')

  @tf.function
  def train_step(self, eye_patches, head_angles, gaze_out):
    with tf.GradientTape as tape:
      pred = self.model(eye_patches, head_angles, training = True)
      loss = self.loss_fn(gaze_out, pred)
    gradients = tape.gradients(loss, self.model.trainable_variables)
    self.optimizer_fn.apply_gradients(zip(gradients, self.model.trainable_variables))
    self.train_loss(loss)

  @tf.function
  def test_step(self, eye_patches, head_angles, gaze_out):
    pred = self.model(eye_patches, head_angles, training = False)
    loss = self.loss_fn(gaze_out, pred)
    self.test_loss(loss)

  def train(self, train_ds, test_ds):
    for epoch in range(self.config['num_epochs']):
      self.train_loss.reset_states()
      self.test_loss.reset_states()

      for input_imgs, input_refs, gaze_out in train_ds:
        self.train_step(input_imgs, input_refs, gaze_out)

      for input_imgs, input_refs, gaze_out in test_ds:
        self.test_step(input_imgs, input_refs, gaze_out)

      # Print status update
      template = 'Epoch {}, Train Loss: {}, Test Loss: {}'
      print(template.format(
          epoch+1,
          self.train_loss.result(),
          self.test_loss.result()
        )
      )
  
  def predict(self, val_ds):
    # TODO Test if this can be done this way
    sol = []
    for eye_patches, head_angles in val_ds:
      sol.append(self.model(eye_patches, head_angles))
    return sol