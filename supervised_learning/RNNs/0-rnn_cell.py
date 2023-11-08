#!/usr/bin/env python3
"""Imports"""
import numpy as np


class RNNCell:
    """RNN Class"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.i_dim = i
        self.h_dim = h
        self.o_dim = o
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """previous hidden state and input data"""
        concat = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, y

    def softmax(self, x):
        """softmax activation"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
