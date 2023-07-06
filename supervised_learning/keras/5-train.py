#!/usr/bin/env python3
"""train"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """By calling this function, you can train your neural
    network model using mini-batch gradient descent on the
    provided data and labels."""
    model = network.fit(x=data, y=labels,
                        batch_size=batch_size, epochs=epochs,
                        verbose=verbose, shuffle=shuffle, validation_data=validation_data)
    return model
