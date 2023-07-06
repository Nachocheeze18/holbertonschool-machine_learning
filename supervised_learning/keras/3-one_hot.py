#!/usr/bin/env python3
"""imports"""

import tensorflow.keras as K


def one_hot(labels, cls=None):
    """takes a label vector and an optional number
    of classes as input and returns a matrix where
    each row represents a label, and the last dimension
    corresponds to the number of classes, with a value of
    1 indicating the presence of that class and 0 otherwise."""
    lbls = K.backend.variable(labels)
    one_hot_mat = K.backend.one_hot(lbls, num_classes=cls)
    return K.backend.eval(one_hot_mat)
