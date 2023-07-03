#!/usr/bin/env python3
"""neural network"""

import tensorflow.keras as K



def build_model(nx, layers, activations, lambtha, keep_prob):
    """Keras library"""
    model = k.models.Sequential()
    for i, (units, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(k.layers.Dense(units, activation=activation,
                                     input_shape=(nx,),
                                     kernel_regularizer=k.regularizers.l2(lambtha)))
        else:
            model.add(k.layers.Dense(units, activation=activation,
                                     kernel_regularizer=k.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(k.layers.Dropout(1 - keep_prob))
    return model


if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
