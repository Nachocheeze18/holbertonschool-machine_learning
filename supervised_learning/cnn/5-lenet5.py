#!/usr/bin/env python3
""""""

import tensorflow.keras as K


def lenet5(X):
    """"""
    c1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer='he_normal')(X)

    p1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(c1)

    c2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer='he_normal')(p1)

    p2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(c2)

    flat = K.layers.Flatten()(p2)

    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer='he_normal')(flat)

    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer='he_normal')(fc1)

    out = K.layers.Dense(units=10,
                            activation='softmax')(fc2)

    model = K.Model(inputs=X, outputs=out)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
