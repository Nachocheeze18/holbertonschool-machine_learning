#!/usr/bin/env python3
"""one_hot"""

import tensorflow.keras as K


def one_hot(labels, cls=None):
    """takes a label vector and an optional number
    of classes as input and returns a matrix where
    each row represents a label, and the last dimension
    corresponds to the number of classes, with a value of
    1 indicating the presence of that class and 0 otherwise."""
    one_hot = K.utils.to_categorical(labels, cls)
    return one_hot
