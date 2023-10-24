import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

def sampling(args):
    """sample latent space points based on the mean"""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    encoder_input = layers.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)
    z_mean = layers.Dense(latent_dims, activation=None)(x)
    z_log_var = layers.Dense(latent_dims, activation=None)(x)
    z = layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])
    encoder = models.Model(encoder_input, [z, z_mean, z_log_var])

    decoder_input = layers.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)
    output = layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = models.Model(decoder_input, output)

    autoencoder_input = layers.Input(shape=(input_dims,))
    z, z_mean, z_log_var = encoder(autoencoder_input)
    decoded = decoder(z)
    auto = models.Model(autoencoder_input, decoded)

    reconstruction_loss = tf.keras.losses.binary_crossentropy(autoencoder_input, decoded)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
