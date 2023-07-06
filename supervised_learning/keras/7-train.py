#!/usr/bin/env python3
"""train model"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    Train a neural network model using mini-batch gradient descent,
    with optional early stopping and learning rate decay."""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch, lr):
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(schedule)
        callbacks.append(lr_decay)

    return network.fit(x=data, y=labels,
                       batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
