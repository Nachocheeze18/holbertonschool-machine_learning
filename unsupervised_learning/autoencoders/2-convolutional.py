#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    encoder_input = layers.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
    latent = layers.Conv2D(latent_dims, (3, 3),
                           activation='relu', padding='same')(x)
    encoder = Model(encoder_input, latent)

    # Decoder
    decoder_input = layers.Input(shape=latent_dims)
    x = decoder_input
    for f in reversed(filters[:-1]):
        x = layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(filters[-1], (3, 3),
                      activation='relu', padding='valid')(x)
    x = layers.UpSampling2D((2, 2))(x)
    output = layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_input, output)

    # Autoencoder
    autoencoder_input = layers.Input(shape=input_dims)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = Model(autoencoder_input, decoded)

    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

