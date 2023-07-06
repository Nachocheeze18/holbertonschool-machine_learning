#!/usr/bin/env python3
"""save and load in json format"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Save a model's configuration in JSON format to a file.
    """
    config = network.to_json()

    with open(filename, 'w') as file:
        file.write(config)


def load_config(filename):
    """
    Load a model with a specific configuration from a file.
    """
    with open(filename, 'r') as file:
        config = file.read()

    network = K.models.model_from_json(config)

    return network
