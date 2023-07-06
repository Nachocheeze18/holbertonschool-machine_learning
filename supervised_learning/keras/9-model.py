#!/usr/bin/env python3
"""save and load"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Save an entire model to a file.
    """
    network.save(filename)


def load_model(filename):
    """
    Load an entire model from a file.
    """
    return K.models.load_model(filename)
