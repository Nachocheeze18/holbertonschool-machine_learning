#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    input_layer = keras.layers.Input(shape=input_dims)
    x = input_layer

    for filter_size in filters:
        x = keras.layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
    latent = keras.layers.Conv2D(latent_dims[-1], (3, 3),
                                 activation='relu', padding='same')(x)
    encoder = keras.models.Model(input_layer, latent)

    latent_input = keras.layers.Input(shape=latent_dims)
    x = latent_input

    for filter_size in reversed(filters[:-1]):
        x = keras.layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    decoder_output = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)

    decoder = keras.models.Model(latent_input, decoder_output, name='decoder')

    autoencoder_input = keras.layers.Input(shape=input_dims)
    latent_representation = encoder(autoencoder_input)
    reconstructed_input = decoder(latent_representation)
    auto = keras.models.Model(autoencoder_input, reconstructed_input, name='autoencoder')

    auto.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

    return encoder, decoder, auto
