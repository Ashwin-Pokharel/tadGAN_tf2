import os
from typing import Tuple
import numpy as np
import tensorflow as tf

def _wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

class TadGAN(tf.keras.Model):
    """
    TadGAN based on the idea of https://arxiv.org/pdf/2009.07769.pdf

    The model can be used for anomaly detection in univariate & multivariate time series based on a GAN network architecture.

    This GAN architecture uses a encoder-generator network in combination with two discriminator networks (critics).
    One for the encoding of the incoming time series and the other to discriminate the learned embedding.

    This implementation uses the new TensorFlow 2 / Keras Subclassing API instead of a simple Python class.
    """
    def __init__(
        self,
        # Time series hyper parameters
        ts_input_shape: Tuple[int] = (100, 1),
        latent_dim: int = 20,
        gradient_penelty_weight: int = 10,
        n_iterations_critic: int = 5,

        # sub network hyper parameters
        encoder_lstm_units: int = 100,
        generator_lstm_units: int = 100,
        generator_output_activation: str = "tanh",
        critic_x_cnn_blocks: int = 4,
        critic_x_cnn_filters: int = 64,
        critic_z_dense_units: int = 100,

        log_all_losses: bool = True,
        print_model_summaries: bool = False
    ):
        super(TadGAN, self).__init__()

        # Parse default variables
        self.latent_dim = latent_dim
        self.ts_input_shape = ts_input_shape
        self.signal_length = ts_input_shape[0]
        self.n_channels = ts_input_shape[1]
        self.gradient_penelty_weight = gradient_penelty_weight
        self.n_iterations_critic = n_iterations_critic
        self.log_all_losses = log_all_losses

        # TadGAN encoder
        self.encoder_lstm_units = encoder_lstm_units
        self.encoder = self._build_encoder(lstm_units=self.encoder_lstm_units)

        # TadGAN generator
        self.generator_lstm_units = generator_lstm_units
        self.generator_act_fn = generator_output_activation
        self.generator = self._build_generator(generator_lstm_units=self.generator_lstm_units, output_activation=generator_output_activation)

        # TadGAN critic x
        self.critic_x_cnn_filters = critic_x_cnn_filters
        self.critic_x_cnn_blocks = critic_x_cnn_blocks
        self.critic_x = self._build_critic_x(n_cnn_filters=self.critic_x_cnn_filters, n_cnn_blocks=self.critic_x_cnn_blocks)

        # TadGAN critic z
        self.critic_z_dense_units = critic_z_dense_units
        self.critic_z = self._build_critic_z(critic_z_dense_units=self.critic_z_dense_units)

        if print_model_summaries:
            print(self.encoder.summary())
            print(self.generator.summary())
            print(self.critic_x.summary())
            print(self.critic_z.summary())

        self.build(input_shape=(None, ts_input_shape[0], ts_input_shape[1]))

    def get_config(self) -> dict:
        """
        Build a config dict for the custom attributes of this subclassing Keras model

        :return: Config as Python dict
        """
        return dict(
            ts_input_shape=self.ts_input_shape,
            latent_dim=self.latent_dim,
            gradient_penelty_weight=self.gradient_penelty_weight,
            n_iterations_critic=self.n_iterations_critic,
            log_all_losses=self.log_all_losses,

            # Hyperparameters
            encoder_lstm_units=self.encoder_lstm_units,
            generator_lstm_units=self.generator_lstm_units,
            critic_x_cnn_filters=self.critic_x_cnn_filters,
            critic_x_cnn_blocks=self.critic_x_cnn_blocks,
            critic_z_dense_units=self.critic_z_dense_units
        )

    def compile(
        self,
        critic_x_optimizer,
        critic_z_optimizer,
        encoder_generator_optimizer,
        critic_x_loss_fn,
        critic_z_loss_fn,
        encoder_generator_loss_fn,
        **kwargs
    ):
        """
        Customized Keras model compilation based on the idea of https://keras.io/examples/generative/wgan_gp/

        :param critic_x_optimizer: Keras optimizer for the critic x model
        :param critic_z_optimizer: Keras optimizer for the critic z model
        :param encoder_generator_optimizer: Keras optimizer for the encoder & generator model
        :param critic_x_loss_fn: Loss function for the critic x model
        :param critic_z_loss_fn: Loss function for the critic z model
        :param encoder_generator_loss_fn: Loss function for the generator & encoder model

        :param kwargs: Additional kwargs forwarded to the super class

        :return: None
        """
        super(TadGAN, self).compile(**kwargs)

        self.critic_x_optimizer = critic_x_optimizer
        self.critic_z_optimizer = critic_z_optimizer
        self.encoder_generator_optimizer = encoder_generator_optimizer

        self.critic_x_loss_fn = critic_x_loss_fn
        self.critic_z_loss_fn = critic_z_loss_fn
        self.encoder_generator_loss_fn = encoder_generator_loss_fn

    def _build_encoder(self, lstm_units: int = 100):
        """
        Build the Encoder subnetwork for the GAN. This model learns the compressed representation of the input
        time series.
        The encoder uses a single layer BI-LSTM network to learn the compressed representation.
        The number of LSTM units can be adjusted.

        :param lstm_units: Number of LSTM units that could be used for the time series encoding

        :return: Encoder model
        """
        x = tf.keras.layers.Input(shape=self.ts_input_shape, name="encoder_input")

        # Encode the sequence and extend its dimensions
        encoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, return_sequences=True))(x)
        encoded = tf.keras.layers.Flatten()(encoded)
        encoded = tf.keras.layers.Dense(units=self.latent_dim, name="latent_encoding")(encoded)
        encoded = tf.keras.layers.Reshape(target_shape=(self.latent_dim, 1), name='output_encoder')(encoded)

        model = tf.keras.Model(inputs=x, outputs=encoded, name="encoder_model")

        return model

    def _build_generator(self, generator_lstm_units: int = 100, output_activation: str = "tanh") -> tf.keras.Model:
        """
        Build the Generator model for the GAN. This model uses the compressed representation of the encoder and
        tries to reconstruct the original time series from it.
        At the moment a two-layer Bi-LSTM network is used for the reconstruction.

        :param generator_lstm_units: Number of LSTM units that should be used for the reconstruction.
        :param output_activation: The final activation of the generator. Choose with respect to the preprocesssing (scaling/normalizing)
                       of the input time series

        :return: Generator model
        """
        x = tf.keras.layers.Input(shape=(self.latent_dim, 1), name="generator_input")

        # Remove additional dimensions from the latent embedding
        decoded = tf.keras.layers.Flatten()(x)

        # Check if the sequence length is a even number (this is required for this model architecture)
        if self.signal_length % 2 == 1:
            raise ValueError(f"The signal length needs to be even (current signal length: {self.signal_length})")

        # Build the first layer of the generator that should be half the size of the sequence length
        half_seq_length = self.signal_length // 2
        decoded = tf.keras.layers.Dense(units=half_seq_length)(decoded)
        decoded = tf.keras.layers.Reshape(target_shape=(half_seq_length, 1))(decoded)

        # Generation of a new time series using LSTM in combination with up sampling
        decoded = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=generator_lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            merge_mode='concat'
        )(decoded)
        decoded = tf.keras.layers.UpSampling1D(2)(decoded)
        decoded = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=generator_lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            merge_mode='concat'
        )(decoded)

        # Rebuild the original time series signal for all channels
        decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_channels))(decoded)
        decoded = tf.keras.layers.Activation(activation=output_activation)(decoded)

        model = tf.keras.Model(inputs=x, outputs=decoded, name="generator_model")

        return model

    def _build_critic_x(self, n_cnn_filters: int = 64, n_cnn_blocks: int = 4) -> tf.keras.Model:
        """
        Build the critic x network that learns to differ between a fake and real input sequence. The classifier uses a
        stack of 1D CNN block (CNN + leaky relu + dropout) and a final fully connected classifier network.

        :return: Critic x model
        """

        x = tf.keras.layers.Input(shape=self.ts_input_shape, name="critic_x_input")

        if n_cnn_blocks < 1:
            raise ValueError(f"The number of CNN blocks needs to be greater than 1 (current value: {n_cnn_blocks})")

        y = tf.keras.layers.Conv1D(filters=n_cnn_filters, kernel_size=5)(x)
        y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
        y = tf.keras.layers.Dropout(rate=0.25)(y)

        if n_cnn_blocks > 1:
            for i in range(n_cnn_blocks - 1):
                y = tf.keras.layers.Conv1D(filters=n_cnn_filters, kernel_size=5)(y)
                y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
                y = tf.keras.layers.Dropout(rate=0.25)(y)

        y = tf.keras.layers.Flatten()(y)
        y = tf.keras.layers.Dense(1)(y)

        model = tf.keras.Model(inputs=x, outputs=y, name="critic_x_model")

        return model

    def _build_critic_z(self, critic_z_dense_units: int = 100) -> tf.keras.Model:
        """
        Build the critic z model that learns to differ between a real and a fake encoding coming from the encoder.
        The network works with a two-layer fully connected network in combination with a leaky RELU activation
        and dropout regularization.

        :param critic_z_dense_units: Number of units for each of the fully connected layers

        :return: Critic z model
        """

        x = tf.keras.layers.Input(shape=(self.latent_dim, 1), name="critic_z_input")
        y = tf.keras.layers.Flatten()(x)

        y = tf.keras.layers.Dense(units=critic_z_dense_units)(y)
        y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
        y = tf.keras.layers.Dropout(rate=0.2)(y)

        y = tf.keras.layers.Dense(units=critic_z_dense_units)(y)
        y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
        y = tf.keras.layers.Dropout(rate=0.2)(y)

        y = tf.keras.layers.Dense(1)(y)

        model = tf.keras.Model(inputs=x, outputs=y, name="critic_z_model")

        return model

    @tf.function
    def critic_x_gradient_penalty(self, batch_size, y_true, y_pred):
        """
        Calculates the gradient penalty.
        """
        alpha = tf.keras.backend.random_uniform((batch_size, 1, 1))
        interpolated = (alpha * y_true) + ((1 - alpha) * y_pred)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic_x(interpolated)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    @tf.function
    def critic_z_gradient_penalty(self, batch_size, y_true, y_pred):
        """
        Calculates the gradient penalty.
        """
        alpha = tf.keras.backend.random_uniform((batch_size, 1, 1))
        interpolated = (alpha * y_true) + ((1 - alpha) * y_pred)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic_z(interpolated)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((1.0 - norm) ** 2)

        return gp

    @tf.function
    def _critic_x_loss(self, x_mb, z, valid, fake, mini_batch_size):
        """
        Do a step forward and calculate the loss on the critic x model

        :param x_mb: Minibatch of input data
        :param z: Minibatch of random noise
        :param valid: Ground truth vector for valid samples
        :param fake: Ground truth vector for fake samples
        :param mini_batch_size:

        :return: A tuple containing the total loss and the three single losses
        """
        # Do a step forward on critic x model and collect gradients
        x_ = self.generator(z)
        fake_x = self.critic_x(x_)
        valid_x = self.critic_x(x_mb)

        # Calculate critic x loss
        critic_x_valid_cost = self.critic_x_loss_fn(y_true=valid, y_pred=valid_x)
        critic_x_fake_cost = self.critic_x_loss_fn(y_true=fake, y_pred=fake_x)
        # TODO: [SMe] Is the mini_batch size still required?
        critic_x_gradient_penalty = self.critic_x_gradient_penalty(mini_batch_size, x_mb, x_)
        critic_x_total_loss = critic_x_valid_cost + critic_x_fake_cost + (critic_x_gradient_penalty * self.gradient_penelty_weight)

        return critic_x_total_loss, critic_x_valid_cost, critic_x_fake_cost, critic_x_gradient_penalty

    @tf.function
    def _critic_z_loss(self, x_mb, z, valid, fake, mini_batch_size):
        """
        Do a step forward and calculate the loss on the critic z model

        :param x_mb: Minibatch of input data
        :param z: Minibatch of random noise
        :param valid: Ground truth vector for valid samples
        :param fake: Ground truth vector for fake samples
        :param mini_batch_size:

        :return: A tuple containing the total loss and the three single losses
        """
        # Do a step forward on critic z model and collect gradients
        z_ = self.encoder(x_mb)
        fake_z = self.critic_z(z_)
        valid_z = self.critic_z(z)

        # Calculate critic z loss
        critic_z_valid_cost = self.critic_z_loss_fn(y_true=valid, y_pred=valid_z)
        critic_z_fake_cost = self.critic_z_loss_fn(y_true=fake, y_pred=fake_z)
        critic_z_gradient_penalty = self.critic_z_gradient_penalty(mini_batch_size, z, z_)
        critic_z_total_loss = critic_z_valid_cost + critic_z_fake_cost + (critic_z_gradient_penalty * self.gradient_penelty_weight)

        return critic_z_total_loss, critic_z_valid_cost, critic_z_fake_cost, critic_z_gradient_penalty

    @tf.function
    def _encoder_generator_loss(self, x_mb, z, valid):
        """
        Do a step forward and calculate the loss on the encoder-generator model

        :param x_mb: Minibatch of input data
        :param z: Minibatch of random noise
        :param valid: Ground truth vector for valid samples

        :return: A tuple containing the total loss and the three single losses
        """
        # Do a step forward on the encoder generator model
        x_gen_ = self.generator(z)
        fake_gen_x = self.critic_x(x_gen_)

        z_gen_ = self.encoder(x_mb)
        x_gen_rec = self.generator(z_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        # Calculate encoder generator loss
        encoder_generator_fake_gen_x_cost = self.encoder_generator_loss_fn(y_true=valid, y_pred=fake_gen_x)
        encoder_generator_fake_gen_z_cost = self.encoder_generator_loss_fn(y_true=valid, y_pred=fake_gen_z)

        # Use simple MSE as reconstruction error
        general_reconstruction_cost = tf.reduce_mean(tf.square((x_mb - x_gen_rec)))
        encoder_generator_total_loss = encoder_generator_fake_gen_x_cost + encoder_generator_fake_gen_z_cost + (10.0 * general_reconstruction_cost)

        return encoder_generator_total_loss, encoder_generator_fake_gen_x_cost, encoder_generator_fake_gen_z_cost, general_reconstruction_cost

    @tf.function
    def train_step(self, X) -> dict:
        """
        Custom training step for this Subclassing API Keras model.
        The shape should be (n_iterations_critic * batch size, n_channels) because the critic networks are trained
        multiple times over the encoder-generator network.

        :param X: Group of mini batches that are used to train the critics and the encoder-generator network
                  Shape: (batch_size, signal_length, n_channels)

        :return: Sub-model losses as los dict
        """
        if isinstance(X, tuple):
            X = X[0]

        batch_size = X.shape[0]
        mini_batch_size = batch_size // self.n_iterations_critic

        # Prepare the ground truth data
        fake = tf.ones((mini_batch_size, 1))
        valid = -tf.ones((mini_batch_size, 1))

        critic_x_loss_steps = []
        critic_z_loss_steps = []

        # Train the critics multiple steps more then the encoder-generator model
        for critic_train_step in range(self.n_iterations_critic):
            z = tf.random.normal(shape=(mini_batch_size, self.latent_dim, 1))
            x_mb = X[critic_train_step * mini_batch_size: (critic_train_step + 1) * mini_batch_size]

            # Optimize step on critic x
            with tf.GradientTape() as tape:
                # Do a step forward on critic x model and collect gradients
                _critic_x_losses = self._critic_x_loss(x_mb, z, valid, fake, mini_batch_size)

            # Backward step with updating critic x weights
            critic_x_gradient = tape.gradient(_critic_x_losses[0], self.critic_x.trainable_variables)
            self.critic_x_optimizer.apply_gradients(zip(critic_x_gradient, self.critic_x.trainable_variables))

            # Collect summaries for logging
            _critic_x_losses = np.array(_critic_x_losses)
            critic_x_loss_steps.append(_critic_x_losses)

            # Optimize step on critic z
            with tf.GradientTape() as tape:
                # Do a step forward on critic z model and collect gradients
                _critic_z_losses = self._critic_z_loss(x_mb, z, valid, fake, mini_batch_size)

            # Backward step with updating critic z weights
            critic_z_gradient = tape.gradient(_critic_z_losses[0], self.critic_z.trainable_variables)
            self.critic_z_optimizer.apply_gradients(zip(critic_z_gradient, self.critic_z.trainable_variables))

            # Collect summaries for logging
            _critic_z_losses = np.array(_critic_z_losses)
            critic_z_loss_steps.append(_critic_z_losses)

        # Optimize step on encoder & generator and collect gradients
        with tf.GradientTape() as tape:
            # Do a step forward on the encoder generator model
            _encoder_generator_losses = self._encoder_generator_loss(x_mb, z, valid)

        # Backward step with updating encoder generator weights
        encoder_generator_gradient = tape.gradient(_encoder_generator_losses, self.encoder.trainable_variables + self.generator.trainable_variables)
        self.encoder_generator_optimizer.apply_gradients(zip(encoder_generator_gradient, self.encoder.trainable_variables + self.generator.trainable_variables))

        # Collect summaries for logging
        critic_x_losses = np.mean(np.array(critic_x_loss_steps), axis=0)
        critic_z_losses = np.mean(np.array(critic_z_loss_steps), axis=0)
        encoder_generator_losses = np.array(_encoder_generator_losses)

        if self.log_all_losses:
            loss_dict = {
                "Cx_total": critic_x_losses[0],
                "Cx_valid": critic_x_losses[1],
                "Cx_fake": critic_x_losses[2],
                "Cx_gp_penalty": critic_x_losses[3],

                "Cz_total": critic_z_losses[0],
                "Cz_valid": critic_z_losses[1],
                "Cz_fake": critic_z_losses[2],
                "Cz_gp_penalty": critic_z_losses[3],

                "EG_total": encoder_generator_losses[0],
                "EG_fake_gen_x": encoder_generator_losses[1],
                "EG_fake_gen_z": encoder_generator_losses[2],
                "G_rec": encoder_generator_losses[3],
            }
        else:
            loss_dict = {
                "Cx_total": critic_x_losses[0],
                "Cz_total": critic_z_losses[0],
                "EG_total": encoder_generator_losses[0]
            }

        return loss_dict

    @tf.function
    def test_step(self, X):
        """
        Custom test step for this Subclassing API Keras model.
        This overrides the default behavior of the model.evaluate() function.

        :param X: Minibatch of time series signals (batch_size, signal_length, n_channels)

        :return: Sub-model losses as los dict
        """

        if isinstance(X, tuple):
            X = X[0]

        batch_size = X.shape[0]

        # Prepare the ground truth data
        fake = tf.ones((batch_size, 1))
        valid = -tf.ones((batch_size, 1))

        z = tf.random.normal(shape=(batch_size, self.latent_dim, 1))

        critic_x_losses = self._critic_x_loss(X, z, valid, fake, batch_size)
        critic_z_losses = self._critic_z_loss(X, z, valid, fake, batch_size)
        encoder_generator_losses = self._encoder_generator_loss(X, z, valid)

        if self.log_all_losses:
            loss_dict = {
                "Cx_total": critic_x_losses[0],
                "Cx_valid": critic_x_losses[1],
                "Cx_fake": critic_x_losses[2],
                "Cx_gp_penalty": critic_x_losses[3],

                "Cz_total": critic_z_losses[0],
                "Cz_valid": critic_z_losses[1],
                "Cz_fake": critic_z_losses[2],
                "Cz_gp_penalty": critic_z_losses[3],

                "EG_total": encoder_generator_losses[0],
                "EG_fake_gen_x": encoder_generator_losses[1],
                "EG_fake_gen_z": encoder_generator_losses[2],
                "G_rec": encoder_generator_losses[3],
            }
        else:
            loss_dict = {
                "Cx_total": critic_x_losses[0],
                "Cz_total": critic_z_losses[0],
                "EG_total": encoder_generator_losses[0]
            }

        return loss_dict

    @tf.function
    def call(self, X, **kwargs) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Default forward step during the inference step of the model (for training and evaluation see train_step() and test_step()).
        This function will be called in the model.predict() function to use the trained sub networks for anomaly detection.

        :param X: Batch of signals that should be analyzed by the model (batch_size, signal_length, n_channels)

        :param kwargs: Additional kwargs forwarded to the super class fit method

        :return: Tuple containing the outputs of the sub networks as numpy arrays:
                 - The reconstructed signals from the generator
                 - The compressed embedding of the time series (latent_dim, 1) from the encoder
                 - The fake/real classification result for the reconstructed time series from the critic x network
                 - The fake/real classification result for the learned embedding from the critic z network
        """
        latent_encoding = self.encoder(X)
        y_hat = self.generator(latent_encoding)
        critic_x = self.critic_x(X)
        critic_z = self.critic_z(latent_encoding)
        return y_hat, latent_encoding, critic_x, critic_z

    def fit(
        self,
        x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
        validation_split=0., validation_data=None, shuffle=True,
        class_weight=None, sample_weight=None,
        initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1,
        max_queue_size=10, workers=1, use_multiprocessing=False
    ):
        """
        Extends the orignal fit method of the keras.Model API with some checks for the train_step batch size requirements.
        See for https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit more details

        :return: Keras model training history dict
        """
        # Adjust batch size to support multiple training steps on the critics
        if not isinstance(validation_data, tf.data.Dataset):
            if (validation_data is not None) and (validation_batch_size is None):
                validation_batch_size = batch_size

        if not isinstance(x, tf.data.Dataset):
            batch_size = batch_size * self.n_iterations_critic

        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                           class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                           validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)

    @staticmethod
    def export_as_keras_model(tadgan_model, export_path: str):
        # Rebuild model graph
        x = tf.keras.layers.Input(shape=tadgan_model.ts_input_shape, name="ts_input")
        latent_encoding = tadgan_model.encoder(x)
        y_hat = tadgan_model.generator(latent_encoding)
        critic_x = tadgan_model.critic_x(x)
        critic_z = tadgan_model.critic_z(latent_encoding)

        # Export model
        standalone_model = tf.keras.models.Model(inputs=x, outputs=[y_hat, latent_encoding, critic_x, critic_z])
        standalone_model.save(export_path)



