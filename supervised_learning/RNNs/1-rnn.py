#!/usr/bin/env python3
"""Imports"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN"""
    t, m, _ = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    o = rnn_cell.Wy.shape[1]
    Y = np.zeros((t, m, o))

    h_t = h_0

    for step in range(t):
        x_t = X[step, :, :]
        h_t, y_t = rnn_cell.forward(h_t, x_t)
        H[step + 1, :, :] = h_t
        Y[step, :, :] = y_t

    return H, Y
