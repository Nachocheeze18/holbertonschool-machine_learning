#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates an autoencoder or a sparse autoencoder depending on lambtha value"""
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu', activity_regularizer=keras.regularizers.l1(lambtha))(x)
    encoder = keras.Model(encoder_input, latent)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, output)

    # Autoencoder
    autoencoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.Model(autoencoder_input, decoded)

    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    # Check conditions for encoder layers
    conditions = ([layer.activation == keras.activations.relu and
                   layer.activation is not None and layer.units
                   is not None for layer in encoder.layers[1:]])
    print(all(conditions))

    return encoder, decoder, auto
