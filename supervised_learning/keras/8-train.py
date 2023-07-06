#!/usr/bin/env python3
"""training func"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Train a neural network model using mini-batch gradient descent,
    with optional early stopping, learning rate decay, and saving the best model.
    """
    callback_list = []

    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience
        )
        callback_list.append(early_stopping_callback)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay_callback = K.callbacks.LearningRateScheduler(
            scheduler, verbose=1
        )
        callback_list.append(lr_decay_callback)

    if save_best and filepath is not None:
        model_checkpoint_callback = K.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True
        )
        callback_list.append(model_checkpoint_callback)

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callback_list
    )
