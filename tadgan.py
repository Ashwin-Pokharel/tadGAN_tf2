#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import keras from tensorflow
import os
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tadgan_example import TadGAN
from tqdm import trange
import sys


# <h1>function for wasserstein loss</h1>
# - described in the [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
# <br>
# - implemented in [tadGAN](https://arxiv.org/pdf/2009.07769.pdf)
# 

# In[2]:


def _wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


# <h1>Define all of the input shapes and parameters</h1>
# 

# In[3]:


seq_len = 100 # sequence length must always be even
ts_input_shape: Tuple[int] = (seq_len, 8)#this is for univariant data , change the shape accorindly ex for multivariant data change to (100,num_features)
latent_dim: int = 20 #latent dimension where encoder and decoder will be trained
gradient_penalty_weight: int = 10#gradient penelty weight for wasserstein loss
n_iterations_critic: int = 5#number of iterations for training the critic per iter for encoder and decoder

# sub network hyper parameters
encoder_lstm_units: int = 100 # number of units in encoder LSTM
generator_lstm_units: int = 100 # number of units in generator LSTM
generator_output_activation: str = "tanh" # activation function for generator output
critic_x_cnn_blocks: int = 4 # number of convolutional blocks in critic x
critic_x_cnn_filters: int = 64 # number of filters in each convolutional block in critic x
critic_z_dense_units: int = 100 # number of units in critic z dense layer

log_all_losses: bool = True
print_model_summaries: bool = True 


# <h1>Make an encoder layer </h1>
#     - Build the Encoder subnetwork for the GAN. This model learns the compressed representation of the input and transforms the timesries sequence into the latent space
#     </br>
#     - The encoder uses a single layer BI-LSTM network to learn the compressed representation.

# In[4]:


def generate_encoder(input_shape: Tuple[int]=(100,1), lstm_units:int = 100, latent_dim:int=20)->tf.keras.Model:
    """
        The number of LSTM units can be adjusted.

        :param lstm_units: Number of LSTM units that could be used for the time series encoding

        :input_shape: Tuple of input shape (batch_size, len_sequence, num_features)

        :return: Encoder model
    """

    input = tf.keras.layers.Input(shape=input_shape , name="encoder_input")
    #create a bi-directional LSTM layer
    encoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, return_sequences=True))(input)
    encoded = tf.keras.layers.Flatten()(encoded)
    encoded = tf.keras.layers.Dense(units=latent_dim, name="latent_encoding")(encoded)
    encoded = tf.keras.layers.Reshape(target_shape=(latent_dim, 1) , name="output_encoder")(encoded)

    model = tf.keras.Model(inputs=input, outputs=encoded, name="encoder")

    return model


# <h1>Make an generator layer </h1>
#     - Build the generator subnetwork for the GAN. This recreates the timeseries sequence from the latent space
#     </br>
#     - The generator  uses a double  layer BI-LSTM network to recreate the timeseries data.

# In[5]:


def generate_generator(latent_shape: Tuple[int], lstm_units:int = 64, activation_function:str="tanh") -> tf.keras.Model:
    """
        The number of LSTM units can be adjusted.

        :param lstm_units: Number of LSTM units that could be used for the time series generation

        :param latent_shape: Shape of the latent encoding

        :param activation_function: final activation layer for the generator 

        :return: Generator model
    """

    input = tf.keras.layers.Input(shape=latent_shape, name="generator_input")
    decoded = tf.keras.layers.Flatten()(input)

    #first layer should be half the size of the sequence
    half_seq_length = seq_len // 2
    decoded = tf.keras.layers.Dense(units=half_seq_length)(decoded)
    decoded = tf.keras.layers.Reshape(target_shape=(half_seq_length, 1))(decoded)  

    # generate a new timeseries using two ltsm layers that have 64 hidden units with upsampling  between them
    decoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=0.2 , recurrent_dropout=0.2), merge_mode="concat")(decoded)
    decoder = tf.keras.layers.UpSampling1D(size=2)(decoder)
    decoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=0.2 , recurrent_dropout=0.2), merge_mode="concat")(decoder)

    #rebuild the original shape of the time series for all signals
    decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(ts_input_shape[1]//n_iterations_critic))(decoder)
    decoder = tf.keras.layers.Activation(activation_function)(decoder)
    return tf.keras.Model(inputs=input, outputs=decoder, name="generator")


# <h1>Make an critic_x  layer </h1>
#     - Build the critic subnetwork for the GAN. This distinguishes between the timeseries sequence and the generated sequence.
#     </br>
#     - The critic uses sequence of 1d convolutional layers to distinguish between the timeseries and generated sequence. and finally a fully connected layer

# In[6]:


def build_critic_x(input_shape ,num_filters: int = 64, num_cnn_blocks: int = 4) -> tf.keras.Model:
    """
        Builds the critic model for the critic_x

        :param num_filters: Number of filters in each convolutional block

        :param num_cnn_blocks: Number of convolutional blocks in the critic

        :return: Critic model
    print(input_shape)
    """
    input = tf.keras.layers.Input(shape=input_shape, name="critic_x_input")
    #create a convolutional layer with num_filters filters
    conv = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=5)(input)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
    conv = tf.keras.layers.Dropout(0.25)(conv)

    for _ in range(num_cnn_blocks):
        conv = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=5, padding="same")(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
        conv = tf.keras.layers.Dropout(0.25)(conv)

    #flatten the output and create a de#nse output layer
    conv = tf.keras.layers.Flatten()(conv)
    conv = tf.keras.layers.Dense(units=1)(conv)

    return tf.keras.Model(inputs=input, outputs=conv, name="critic_x")


# <h1>Make an critic_z layer </h1>
#     - Build the critic subnetwork for the GAN. This distinguishes between the timeseries sequence and the generated sequence.
#     </br>
#     - The critic uses two fully connected layers to  distinguish between the real encoded sequence and fake encoding sequence.

# In[7]:


def build_critic_z(latent_space_dim: Tuple[int , int] ,num_dense_units: int = 100)->tf.keras.Model:
    """
        Builds the critic model for critic_z

        :param latent_space_dim: shaoe of the latent space

        :param num_dense_units: Number of units in the dense layer

        :return: Critic model
    """

    input = tf.keras.layers.Input(shape=latent_space_dim, name="critic_z_input")

    dense = tf.keras.layers.Flatten()(input)
    dense = tf.keras.layers.Dense(units=num_dense_units)(input)
    dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
    dense = tf.keras.layers.Dropout(0.25)(dense)

    dense = tf.keras.layers.Dense(units=num_dense_units)(dense)
    dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
    dense = tf.keras.layers.Dropout(0.25)(dense)
    
    dense = tf.keras.layers.Dense(units=1)(dense)

    model = tf.keras.Model(inputs=input, outputs=dense, name="critic_z")
    return model


# <h1>calculate gradient penalty </h1>
#     <h2></h2>
#     - The gradient penalty is used to ensure that the critic is not too sure about the discriminator's ability to distinguish between the real and fake sequences.
#     <br>
#     - This reguralizations is used to reduce the risk of gradient exploding.

# In[8]:


@tf.function
def critic_x_gradient_penalty(critic_x , batch_size, y_true, y_pred):
    """
    Calculates the gradient penalty.
    """
    alpha = tf.keras.backend.random_uniform((batch_size, 1, 1))
    interpolated = (alpha * y_true) + ((1 - alpha) * y_pred)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = critic_x(interpolated)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)

    return gp

@tf.function
def critic_z_gradient_penalty(critic_z, batch_size, y_true, y_pred):
    """
    Calculates the gradient penalty.
    """
    alpha = tf.keras.backend.random_uniform((batch_size, 1, 1))
    interpolated = (alpha * y_true) + ((1 - alpha) * y_pred)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = critic_z(interpolated)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((1.0 - norm) ** 2)


# <h1>Calculate loss function </h1>
# 
# - do loss calculations for the critics , encoder and generator

# In[9]:


@tf.function
def critic_x_loss(generator,critic_x, critic_x_loss_fn ,gradient_penelty_weight ,   x_mb, z, valid, fake, mini_batch_size):
    """
    Do a step forward and calculate the loss on the critic x model

    :param generator: Generator model
    :param critic_x: Critic x model
    :param critic_x_loss_fn: Critic x loss function
    :param gradient_penalty_weight: Weight of the gradient penalty
    :param x_mb: Minibatch of input data
    :param z: Minibatch of random noise
    :param valid: Ground truth vector for valid samples
    :param fake: Ground truth vector for fake samples
    :param mini_batch_size:

    :return: A tuple containing the total loss and the three single losses
    """
    # Do a step forward on critic x model and 
    x_ = generator(z)
    
    fake_x = critic_x(x_)
    valid_x = critic_x(x_mb)

    # Calculate critic x loss
    critic_x_valid_cost = critic_x_loss_fn(y_true=valid, y_pred=valid_x)
    critic_x_fake_cost = critic_x_loss_fn(y_true=fake, y_pred=fake_x)
    # TODO: [SMe] Is the mini_batch size still required?
    critic_x_gradient_penalty = critic_x_gradient_penalty(mini_batch_size, x_mb, x_)
    critic_x_total_loss = critic_x_valid_cost + critic_x_fake_cost + (critic_x_gradient_penalty * gradient_penelty_weight)

    return critic_x_total_loss, critic_x_valid_cost, critic_x_fake_cost, critic_x_gradient_penalty

@tf.function
def critic_z_loss(encoder , critic_z, critic_z_loss_fn, gradient_penelty_weight,x_mb, z, valid, fake, mini_batch_size):
    """
    Do a step forward and calculate the loss on the critic z model

    :param encoder: Encoder model
    :param critic_z: Critic z model
    :param critic_z_loss_fn: Critic z loss function
    :param gradient_penalty_weight: Weight of the gradient penalty
    :param x_mb: Minibatch of input data
    :param z: Minibatch of random noise
    :param valid: Ground truth vector for valid samples
    :param fake: Ground truth vector for fake samples
    :param mini_batch_size:

    :return: A tuple containing the total loss and the three single losses
    """
    # Do a step forward on critic z model and collect gradients
    z_ = encoder(x_mb)
    fake_z = critic_z(z_)
    valid_z = critic_z(z)

    # Calculate critic z loss
    critic_z_valid_cost = critic_z_loss_fn(y_true=valid, y_pred=valid_z)
    critic_z_fake_cost = critic_z_loss_fn(y_true=fake, y_pred=fake_z)
    critic_z_gradient_penalty = critic_z_gradient_penalty(mini_batch_size, z, z_)
    critic_z_total_loss = critic_z_valid_cost + critic_z_fake_cost + (critic_z_gradient_penalty * gradient_penelty_weight)
    return critic_z_total_loss, critic_z_valid_cost, critic_z_fake_cost, critic_z_gradient_penalty


@tf.function
def encoder_generator_loss(generator , encoder , critic_x , critic_z , encoder_generator_loss_fn , x_mb, z, valid):
    """
    Do a step forward and calculate the loss on the encoder-generator model

    :param generator: Generator model
    :param encoder: Encoder model
    :param critic_x: Critic x model
    :param critic_z: Critic z model
    :param encoder_generator_loss_fn: Encoder-generator loss function
    :param x_mb: Minibatch of input data
    :param z: Minibatch of random noise
    :param valid: Ground truth vector for valid samples

    :return:   Do ale containing the total loss and the three s
    print(x_mb.shape)ingle losses
    """
    # Do a step forward on the encode generator model
    x_gen_ = generator(z)
    fake_gen_x = critic_x(x_gen_)

    z_gen_ = encoder(x_mb)
    x_gen_rec = generator(z_gen_)
    fake_gen_z = critic_z(z_gen_)

    # Calculate encoder generator loss
    encoder_generator_fake_gen_x_cost = encoder_generator_loss_fn(y_true=valid, y_pred=fake_gen_x)
    encoder_generator_fake_gen_z_cost = encoder_generator_loss_fn(y_true=valid, y_pred=fake_gen_z)

    # Use simple MSE as reconstruction error
    general_reconstruction_cost = tf.reduce_mean(tf.square((x_mb - x_gen_rec)))
    encoder_generator_total_loss = encoder_generator_fake_gen_x_cost + encoder_generator_fake_gen_z_cost + (10.0 * general_reconstruction_cost)

    return encoder_generator_total_loss, encoder_generator_fake_gen_x_cost, encoder_generator_fake_gen_z_cost, general_reconstruction_cost
    


# <h1>Training step for the model </h1>
# - do training step for the critics , encoder and generator
# 

# In[10]:


@tf.function
def train_step(input, n_iterations_critic,encoder, generator,critic_x , critic_z) -> dict:
    """
    Custom training step for this Subclassing API Keras model.
    The shape should be (n_iterations_critic * batch size, n_channels) because the critic networks are trained
    multiple times over the encoder-generator network.

    :param X: Group of mini batches that are used to train the critics and the encoder-generator network
                Shape: (batch_size, signal_length, n_channels)
    :param n_iterations_critic: Number of iterations to train the critic network
    :param critic_x_loss: Critic x loss function
    :param critic_z_loss: Critic z loss function
    :param encoder_generator_loss: Encoder-generator loss function
    :param critic_x_optimizer: Critic x optimizer
    :param critic_z_optimizer: Critic z optimizer


    :return: Sub-model losses as los dict
    """
    # Get the input data
    X = input[0] if isinstance(input , tuple) else input
    batch_size = X.shape[0]
    minibatch_size = batch_size//n_iterations_critic
    critic_x_loss_fn = tf.keras.losses.MeanSquaredError()
    critic_z_loss_fn = tf.keras.losses.MeanSquaredError()
    encoder_generator_loss_fn = tf.keras.losses.MeanSquaredError()
    encoder_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_x_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_z_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    #prepare ground truth data
    valid = tf.ones((minibatch_size, 1))
    fake = -tf.ones((minibatch_size, 1))

    critic_x_loss_steps = []
    critic_z_loss_steps = []

    for critic_train_steps in range(n_iterations_critic):
        z = tf.random.normal((minibatch_size, latent_dim, 1))
        x_mb = X[critic_train_steps * minibatch_size:(critic_train_steps + 1) * minibatch_size]
        x_mb.reshape((1 , x_mb.shape[1] , x_mb.shape[0]))
        print("shape of valid input {0}".format(x_mb.shape))

        #optimize step on critic x mode                    
        with tf.GradientTape() as tape:
            _critic_x_losses = critic_x_loss(generator, critic_x, critic_x_loss_fn, gradient_penalty_weight , x_mb, z, valid, fake, minibatch_size)

        #backward step on critic x model
        critic_x_gradient = tape.gradient(_critic_x_losses[0], critic_x.trainable_variables)
        critic_x_optimizer.apply_gradients(zip(critic_x_gradient, critic_x.trainable_variables))

        _critic_x_losses = np.array(_critic_x_losses)
        critic_x_loss_steps.append(_critic_x_losses)


        #optimize step on critic z model
        with tf.GradientTape() as tape:
            _critic_z_losses = critic_z_loss(encoder , critic_z , critic_z_loss_fn, gradient_penalty_weight , x_mb, z, valid, fake, minibatch_size)

        #backward step on critic z model
        critic_z_gradient = tape.gradient(_critic_z_losses[0], critic_z.trainable_variables)
        critic_z_optimizer.apply_gradients(zip(critic_z_gradient, critic_z.trainable_variables))

        _critic_z_losses = np.array(_critic_z_losses)
        critic_z_loss_steps.append(_critic_z_losses)

    #optimize step on encoder-generator model
    with tf.GradientTape() as tape:
        #step forward for the generator model
        _encoder_generator_losses = encoder_generator_loss(generator, encoder, critic_x, critic_z, encoder_generator_loss_fn, X, z, valid)

    #backward step on encoder-generator model
    encoder_generator_gradient = tape.gradient(_encoder_generator_losses, encoder.trainable_variables +  generator.trainable_variables)
    encoder_generator_optimizer.apply_gradients(zip(encoder_generator_gradient, encoder.trainable_variables + generator.trainable_variables))


    critic_x_losses = np.mean(np.array(critic_x_loss_steps), axis=0)
    critic_z_losses = np.mean(np.array(critic_z_loss_steps), axis=0)
    encoder_generator_losses = np.array(_encoder_generator_losses)

    if log_all_losses:
        return {
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
        return {
            "Cx_total": critic_x_losses[0],
            "Cz_total": critic_z_losses[0],
            "EG_total": encoder_generator_losses[0]
        }




# In[11]:


@tf.function
def test_step(input,encoder , generator , critic_x , critic_z, gradient_penalty_weight):
    """
    
    test step for model
    overrides model.evaluate

    :param input: minibatch of the time series signals (batce_size , signal_length , n_channels)
    :param critic_x_loss: loss function for critic x
    :param critic_z_loss: loss function for critic z
    :param encoder_generator_loss: loss function for encoder and generator
    :param graident_penalty_weight: penalty weight for gradient

    :return: sub-model losses as loss dict

    """

    if isinstance(input, tuple):
            input = input[0]

    batch_size = input.shape[0]

    fake = tf.ones((batch_size , 1))
    valid = tf.ones((batch_size , 1))


    critic_x_loss_fn = tf.keras.losses.MeanSquaredError()
    critic_z_loss_fn = tf.keras.losses.MeanSquaredError()
    encoder_generator_loss_fn = tf.keras.losses.MeanSquaredError()
    encoder_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_x_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_z_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    z = tf.random.normal(shape=(batch_size , latent_dim , 1))

    critic_x_losses = critic_x_loss(generator,critic_x, critic_x_loss_fn ,gradient_penalty_weight , input , z, valid, fake, batch_size)
    critic_z_losses = critic_z_loss(encoder , critic_z, critic_z_loss_fn, gradient_penalty_weight, input , z, valid, fake, batch_size)
    encoder_generator_losses = encoder_generator_loss(generator , encoder , critic_x , critic_z , encoder_generator_loss_fn , input , z, valid)


    if log_all_losses:
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



# In[12]:


@tf.function
def call( input, encoder , generator , critic_x , critic_z, **kwargs) -> Tuple[np.array, np.array, np.array, np.array]:
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
    X = input[0] if isinstance(input , tuple) else input
    latent_encoding = encoder(X)
    y_hat = generator(latent_encoding)
    critic_x = critic_x(X)
    critic_z = critic_z(latent_encoding)
    return y_hat, latent_encoding, critic_x, critic_z


# <h1>Generate the models to be used during training</h1>

# In[13]:


encoder = generate_encoder(ts_input_shape , lstm_units=100 , latent_dim=latent_dim)
generator = generate_generator((latent_dim , 1) , lstm_units=100)
critic_x = build_critic_x(ts_input_shape)
critic_z = build_critic_z((latent_dim , 1)) 

print(encoder.summary())
print(generator.summary())
print(critic_x.summary())
print(critic_z.summary())

encoder_checkpoint = "training_checkpoints/encoder-{epoch:04d}.cpkt"
generator_checkpoint = "training_checkpoints/generator-{epoch:04d}.cpkt"
critic_x_checkpoint = "training_checkpoints/ciritic_x-{epoch:04d}.cpkt"
critic_z_checkpoint = "training_checkpoints/critic_z-{epoch:04d}.cpkt"


# <h1>Get the Dataset</h1>
# - create a tf batch dataset out of the CSV
# <br>
# - data should not be shuffeled as the data's sequential information needs to be preserved

# In[16]:


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


print("fetching dataset") 
dataset = tf.data.experimental.make_csv_dataset("fan_speed_vibration_without_duration.csv", batch_size = seq_len * n_iterations_critic ,shuffle=False)
print("fetched dataset , transforming dataset for input")
dataset = tf.data.Dataset.from_tensor_slices(list(dataset))
print("shape of dataset: {0}".format(dataset.shape))
sys.exit()
dataset = dataset.shuffle(buffer_size=dataset.size[0])
print("done with dataset tranformation")
#dataset = dataset.shuffle(buffer_size=1024).batch(seq_len)
#tf.data.experimental.CsvDataset("fan_speed_vibration_without_duration.csv", [tf.float32 , tf.float32 , tf.float32 . tf.float32, tf.float32 , tf.float32 , tf.float32 . tf.float32])

#dataset = pd.read_csv("fan_speed_vibration_without_duration.csv")


# <h1>Training the model</h1>
# 

# In[ ]:


epoch = 20
save_dir = "models/"
print("starting training process")
with trange(epoch , position=0 , unit="epoch") as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            for step , data in enumerate(dataset):
                data = list(data.values())
                data = np.array(data)
                old_shape = data.shape
                data = data.reshape(old_shape[1] , old_shape[0])
                loss_dict = train_step(data, n_iterations_critic , encoder , generator , critic_x , critic_z)
                pbar.set_postfix(loss_dict , refresh=True)
                print("first step is done")
            if epoch % 5 == 0:
                encoder.save_weights(encoder_checkpoint.format(epoch))
                generator.save_weights(generator_checkpoint.format(epoch))
                critic_x.save_weights(critic_x_checkpoint.format(epoch))
                critic_z.save_weights(critic_z_checkpoint.format(epoch))
                
        
        encoder.save("models/encoder")
        generator.save("models/generator")
        critic_x.save("models/critic_x")
        critic_z.save("model/critic_z")
'''
tadGan = TadGAN()
print(tadGan.encoder.summary())
print(tadGan.generator.summary())
print(tadGan.critic_x.summary())
print(tadGan.critic_z.summary())
tadGan.summary()
'''
            


# In[ ]:




