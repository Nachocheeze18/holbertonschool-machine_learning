#!/usr/bin/env python3
"""imports"""

Import tensorflow.keras as K


def one_hot(labels, cls=None):
    """takes a label vector and an optional number
    of classes as input and returns a matrix where
    each row represents a label, and the last dimension
    corresponds to the number of classes, with a value of
    1 indicating the presence of that class and 0 otherwise."""
    if cls is None:
        max_label = max(labels)
        cls = max_label + 1
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=cls)
    return one_hot_matrix
