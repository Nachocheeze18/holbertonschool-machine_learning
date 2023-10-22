#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates an autoencoder or a sparse autoencoder depending on lambtha value"""

    input_data = keras.layers.Input(shape=(input_dims,))
    
    encoded = input_data
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    if lambtha:
        encoded = keras.layers.Dense(latent_dims, activation='relu',
                    activity_regularizer=keras.regularizers.l1(lambtha))(encoded)

    decoded = encoded
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    encoder = keras.models.Model(input_data, encoded)

    decoder = keras.models.Model(encoded, decoded)

    auto = keras.models.Model(input_data, decoder(encoded))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
