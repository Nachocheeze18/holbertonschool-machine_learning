#!/usr/bin/env python3
"""save and load in json format"""

import tensorflow.keras as K
import json


def save_config(network, filename):
    """
    Save a model's configuration in JSON format to a file.
    """
    config = network.get_config()
    config_json = json.dumps(config, indent=4)

    with open(filename, 'w') as file:
        file.write(config_json)


def load_config(filename):
    """
    Load a model with a specific configuration from a file.
    """
    with open(filename, 'r') as file:
        config_json = file.read()

    config = json.loads(config_json)
    network = K.models.Model.from_config(config)

    return network
