#!/usr/bin/env python3
""" builds the ResNet-50 architecture as described
in Deep Residual Learning for Image Recognition (2015):
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds a ResNet-50 architecture for image recognition.
    It constructs a deep neural network model with convolutional layers,
    batch normalization, rectified linear activation, and blocks (identity
    and projection) as described in the "Deep Residual Learning for Image
    Recognition" paper. The function returns the constructed ResNet-50
    model using Keras."""
    input = (224, 224, 3)

    input_layer = K.Input(shape=input)

    x = K.layers.Conv2D(filters=64,
                        kernel_size=(7, 7),
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal'
                        )(input_layer)

    x = K.layers.BatchNormalization(axis=3)(x)

    x= K.layers.ReLU()(x)

    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=2,
                              padding='same'
                              )(x)

    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = projection_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = projection_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = projection_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=1,
                                  padding='valid'
                                  )(x)

    x = K.layers.Dense(units=1000,
                       activation='softmax'
                       )(x)

    return K.Model(inputs=input, outputs=x)