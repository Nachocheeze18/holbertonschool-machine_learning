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

    c7 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         strides=2,
                         padding='same',
                         activation='relu'
                         )(inputs)

    Max3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=2,
                                 padding='same'
                                 )(c7)

    c1 = K.layers.Conv2D(filters=64,
                         kernel_size=(1, 1),
                         padding='valid',
                         activation='relu'
                         )(Max3)

    c3 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         strides=1,
                         padding='same',
                         activation='relu'
                         )(c1)

    Max3x3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                   strides=2,
                                   padding='same'
                                   )(c3)

    i_layer_0 = inception_block(Max3x3,
                                [64, 96, 128, 16, 32, 32])

    i_layer_1 = inception_block(i_layer_0,
                                [128, 128, 192, 32, 96, 64])

    Max3x3_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=2,
                                     padding='same'
                                     )(i_layer_1)

    i_layer_2 = inception_block(Max3x3_1,
                                [192, 96, 208, 16, 48, 64])

    i_layer_3 = inception_block(i_layer_2,
                                [160, 112, 224, 24, 64, 64])

    i_layer_4 = inception_block(i_layer_3,
                                [128, 128, 256, 24, 64, 64])

    i_layer_5 = inception_block(i_layer_4,
                                [112, 144, 288, 32, 64, 64])

    i_layer_6 = inception_block(i_layer_5,
                                [256, 160, 320, 32, 128, 128])

    Max3x3_2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=2,
                                     padding='same'
                                     )(i_layer_6)

    i_layer_7 = inception_block(Max3x3_2,
                                [256, 160, 320, 32, 128, 128])

    i_layer_8 = inception_block(i_layer_7,
                                [384, 192, 384, 48, 128, 128])

    AvgPool7 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=1,
                                         padding='valid'
                                         )(i_layer_8)

    dropout = K.layers.Dropout(.4)(AvgPool7)

    softmax = K.layers.Dense(units=1000,
                             activation='softmax'
                             )(dropout)

    model = K.Model(inputs=inputs, outputs=softmax)

    return model
