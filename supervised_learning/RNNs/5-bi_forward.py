#!/usr/bin/env python3
"""Imports"""
import numpy as np


class BidirectionalCell:
    """Class that represents a bidirectional cell"""
    def __init__(self, i, h, o):
        """Initialize the cell"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Compute softmax values"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        concat_for = np.concatenate((h_prev, x_t), axis=1)
        h_next_for = np.tanh(np.dot(concat_for, self.Whf) + self.bhf)

        return h_next_for

    def backward(self, h_next, x_t):
        """Performs backward probagation"""
        concat_back = np.concatenate((h_next, x_t), axis=1)
        h_next_back = np.tanh(np.dot(concat_back, self.Whb) + self.bhb)

        return h_next_back
