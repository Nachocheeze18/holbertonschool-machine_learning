#!/usr/binj/env python3
"""imports"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation for a bidirectional RNN."""
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t, m, 2 * h))
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    h_for = h_0
    for step in range(t):
        x_t = X[step, :, :]
        h_for = bi_cell.forward(h_for, x_t)
        H[step, :, :h] = h_for

    h_back = h_t
    for step in range(t - 1, -1, -1):
        x_t = X[step, :, :]
        h_back = bi_cell.backward(h_back, x_t)
        H[step, :, h:] = h_back

    Y = bi_cell.output(H)

    return H, Y
