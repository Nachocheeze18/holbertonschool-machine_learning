#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    input_img = keras.layers.Input(shape=input_dims)
    x = input_img
    for filter in filters:
        x = keras.layers.Conv2D(filter, (3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.models.Model(input_img, x, name='encoder')

    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input

    x = keras.layers.Conv2D(filters[2], (3, 3),
                            activation='relu', padding='same')(x)

    x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(filters[1], (3, 3),
                            activation='relu', padding='same')(x)

    x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(filters[0], (3, 3),
                            activation='relu', padding='valid')(x)

    x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(1, (3, 3),
                            activation='sigmoid', padding='same')(x)

    decoder = keras.models.Model(decoder_input, x, name='decoder')

    encoded_output = encoder(input_img)
    decoded_output = decoder(encoded_output)
    autoencoder = keras.models.Model(input_img,
                                     decoded_output, name='autoencoder')

    autoencoder.compile(optimizer='adam',
                        loss='binary_crossentropy')

    return encoder, decoder, autoencoder
