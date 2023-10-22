#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    input_img = keras.Input(shape=input_dims)
    encoded = input_img
    for filter in filters:
        encoded = keras.layers.Conv2D(filter,
                                      (3, 3),
                                      activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2),
                                            padding='same')(encoded)
    encoder = keras.Model(input_img, encoded, name='encoder')

    decoder_img = keras.Input(shape=latent_dims)
    decoded = decoder_img

    decoded = keras.layers.Conv2D(filters[2],
                                  (3, 3),
                                  activation='relu',
                                  padding='same')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(filters[1],
                                  (3, 3),
                                  activation='relu',
                                  padding='same')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(filters[0],
                                  (3, 3),
                                  activation='relu',
                                  padding='valid')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    # print("Exited loop")
    decoded = keras.layers.Conv2D(1,
                                  (3, 3),
                                  activation='sigmoid',
                                  padding='same')(decoded)
    decoder = keras.Model(decoder_img,
                          decoded,
                          name='decoder')

    # Bring it all together
    encoded_output = encoder(input_img)
    decoded_output = decoder(encoded_output)
    autoencoder = keras.Model(input_img,
                              decoded_output,
                              name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
