#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    input_layers = keras.layers.Input(shape=input_dims)
    x = input_layers
    for filter_size in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    latent = keras.layers.Conv2D(latent_dims[-1], (3, 3),
                                 activation='relu', padding='same')(x)
    encoder = keras.models.Model(input_layers, latent)

    # Decoder
    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input
    for filter_size in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[-1], (3, 3),
                            activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    decoder = keras.models.Model(decoder_input, output)

    # Autoencoder
    autoencoder_input = keras.layers.Input(shape=input_dims)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(autoencoder_input, decoded)

    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
