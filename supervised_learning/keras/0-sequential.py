#!/usr/bin/env python3
"""neural network"""

import tensorflow.keras as K



def build_model(nx, layers, activations, lambtha, keep_prob):
    """Keras library"""
    model = k.models.Sequential()
    for i in range(len(layers)):
        kwargs = {'activation': activations[i]}
        if i == 0:
            kwargs['input_shape'] = (nx,)
            kwargs['kernel_regularizer'] = k.regularizers.l2(lambtha)
        else:
            kwargs['kernel_regularizer'] = k.regularizers.l2(lambtha)
        model.add(k.layers.Dense(layers[i], **kwargs))
        
        if i < len(layers) - 1:
            model.add(k.layers.Dropout(1 - keep_prob))
    return model