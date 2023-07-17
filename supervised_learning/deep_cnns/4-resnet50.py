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
    inputs = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    c1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=2,
                            padding='same',
                            kernel_initializer=init
                            )(inputs)

    b_1 = K.layers.BatchNormalization(axis=3)(c1)

    R_1 = K.layers.ReLU()(b_1)

    Max1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=2,
                                     padding='same'
                                     )(R_1)

    c2_a = projection_block(Max1, [64, 64, 256], s=1)
    c2_b = identity_block(c2_a, [64, 64, 256])
    c2_c = identity_block(c2_b, [64, 64, 256])

    c3_a = projection_block(c2_c, [128, 128, 512])
    c3_b = identity_block(c3_a, [128, 128, 512])
    c3_c = identity_block(c3_b, [128, 128, 512])
    c3_d = identity_block(c3_c, [128, 128, 512])

    c4_a = projection_block(c3_d, [256, 256, 1024])
    c4_b = identity_block(c4_a, [256, 256, 1024])
    c4_c = identity_block(c4_b, [256, 256, 1024])
    c4_d = identity_block(c4_c, [256, 256, 1024])
    c4_e = identity_block(c4_d, [256, 256, 1024])
    c4_f = identity_block(c4_e, [256, 256, 1024])

    c5_a = projection_block(c4_f, [512, 512, 2048])
    c5_b = identity_block(c5_a, [512, 512, 2048])
    c5_c = identity_block(c5_b, [512, 512, 2048])

    AvgPool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=1,
                                        padding='valid'
                                        )(c5_c)

    softmax = K.layers.Dense(units=1000,
                             activation='softmax'
                             )(AvgPool)

    return K.Model(inputs=inputs, outputs=softmax)