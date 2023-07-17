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

    z = K.layers.Conv2D(64, (7, 7), strides=(
        2, 2), padding='same',
        kernel_initializer='he_normal')(input_layer)

    z = K.layers.BatchNormalization(axis=-1)(z)

    z = K.layers.Activation('relu')(z)

    z = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(z)

    z = projection_block(z, filters=(64, 64, 256), s=1)
    z = identity_block(z, filters=(64, 64, 256))
    z = identity_block(z, filters=(64, 64, 256))

    z = projection_block(z, filters=(128, 128, 512))
    z = identity_block(z, filters=(128, 128, 512))
    z = identity_block(z, filters=(128, 128, 512))
    z = identity_block(z, filters=(128, 128, 512))

    z = projection_block(z, filters=(256, 256, 1024))
    z = identity_block(z, filters=(256, 256, 1024))
    z = identity_block(z, filters=(256, 256, 1024))
    z = identity_block(z, filters=(256, 256, 1024))
    z = identity_block(z, filters=(256, 256, 1024))

    z = projection_block(z, filters=(512, 512, 2048))
    z = identity_block(z, filters=(512, 512, 2048))
    z = identity_block(z, filters=(512, 512, 2048))

    z = K.layers.GlobalAveragePooling2D()(z)
    z = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(z)

    model = K.Model(inputs=input_layer, outputs=z)

    return model
