#!/usr/bin/env python3
"""Imports"""
import numpy as np


class GRUCell:
    """represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """Initializes a GRU cell with given dimensions."""
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)

        z_t = self.sigmoid(np.dot(concat, self.Wz) + self.bz)

        r_t = self.sigmoid(np.dot(concat, self.Wr) + self.br)

        concat_res = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_res, self.Wh) + self.bh)

        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
