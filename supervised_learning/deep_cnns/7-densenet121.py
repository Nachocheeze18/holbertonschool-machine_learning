#!/usr/bin/env python3
""" builds the DenseNet-121 architecture as
described in Densely Connected Convolutional Networks:
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """the code defines a function that constructs the DenseNet-121
    architecture with the specified number of filters, growth rate,
    and compression factor."""

    nb_filters = 2 * growth_rate

    init = K.initializers.he_normal()

    inputs = K.Input(shape=(224, 224, 3))

    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=init
    )(x)

    x = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(x)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 6)

    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)

    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)

    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding='valid'
    )(x)

    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init
    )(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    return model
