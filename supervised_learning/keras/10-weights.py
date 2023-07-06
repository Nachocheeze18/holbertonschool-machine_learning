#!/usr/bin/env python3
"""save and load files weight"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Save a model's weights to a file.
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load a model's weights from a file.
    """
    network.load_weights(filename)
