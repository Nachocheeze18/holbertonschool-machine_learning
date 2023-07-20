#!/usr/bin/env python3
"""
builds the inception network as
described in Going Deeper with Convolutions (2014):
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    the code builds the Inception network architecture
    with its various convolutional, pooling, and inception
    block layers, culminating in a classification output layer.
    """
    inputs = K.Input(shape=(224, 224, 3))

    z = K.layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='same',
                        activation='relu',
                        kernel_initializer='he_normal')(inputs)

    z = K.layers.MaxPooling2D((3, 3),
                              strides=(2, 2),
                              padding='same')(z)

    z = K.layers.Conv2D(192,
                        (3, 3),
                        padding='same',
                        activation='relu',
                        kernel_initializer='he_normal')(z)

    z = K.layers.MaxPooling2D((3, 3),
                              strides=(2, 2),
                              padding='same')(z)

    z = inception_block(x, [64, 96, 128, 16, 32, 32])
    z = inception_block(x, [128, 128, 192, 32, 96, 64])

    z = K.layers.MaxPool2D((3, 3),
                           strides=(2, 2),
                           padding='same')(z)

    z = inception_block(z, [192, 96, 208, 16, 48, 64])
    z = inception_block(z, [160, 112, 224, 24, 64, 64])
    z = inception_block(z, [128, 128, 256, 24, 64, 64])
    z = inception_block(z, [112, 144, 288, 32, 64, 64])
    z = inception_block(z, [256, 160, 320, 32, 128, 128])

    z = K.layers.MaxPool2D((3, 3),
                           strides=(2, 2),
                           padding='same')(z)

    z = inception_block(x, [256, 160, 320, 32, 128, 128])
    z = inception_block(x, [384, 192, 384, 48, 128, 128])

    z = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(7, 7),
                                  padding='valid')(z)

    z = K.layers.Dropout(0.4)(z)
    z = K.layers.Dense(1000,
                       activation='softmax',
                       kernel_initializer='he_normal')(z)

    model = K.Model(inputs, z)

    return model
