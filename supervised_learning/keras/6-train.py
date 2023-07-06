#!/usr/bin/env python3
"""train"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Train a neural network model using mini-batch gradient descent,
    with optional early stopping.

    network: Model to train.
    data: Input data.
    labels: Target labels.
    batch_size: Size of the batch used for mini-batch gradient descent.
    epochs: Number of passes through the data for training.
    validation_data: Optional validation data.
    early_stopping: Boolean indicating whether early stopping should be used.
    patience: Patience used for early stopping.
    verbose: Boolean that determines if the output should be printed during training.
    shuffle: Boolean that determines whether to shuffle the batches every epoch.
    Returns: The trained model.
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    return network.fit(x=data, y=labels,
                       batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
