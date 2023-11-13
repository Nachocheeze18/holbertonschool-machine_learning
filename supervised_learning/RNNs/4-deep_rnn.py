#!/usr/bin/env python3
"""Imports"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep RNN"""
    l, m, h = h_0.shape
    t, _, i = X.shape
    z = rnn_cells[-1].Wy.shape[1]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, z))
    H[0] = h_0

    for step in range(t):
        h_prev = X[step]
        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(H[step, layer], h_prev)
            H[step + 1, layer] = h_next
            h_prev = h_next

        Y[step] = y

    return H, Y
