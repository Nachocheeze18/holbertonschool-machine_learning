#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an autoencoder"""
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for num_units in hidden_layers:
        x = keras.layers.Dense(num_units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_input, latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for num_units in reversed(hidden_layers):
        x = keras.layers.Dense(num_units, activation='relu')(x)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, output)

    autoencoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.Model(autoencoder_input, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
