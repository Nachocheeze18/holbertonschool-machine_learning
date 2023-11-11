#!/usr/bin/env python3
"""Imports"""
import numpy as np


class LSTMCell:
    """represents an LSTM unit"""
    def __init__(self, i, h, o):
        """Initializes an LSTM cell with given dimensions."""
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))


    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step."""
        # Concatenate previous hidden state and input data
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = sigmoid(np.dot(concat, self.Wf) + self.bf)

        u = sigmoid(np.dot(concat, self.Wu) + self.bu)

        c_hat = np.tanh(np.dot(concat, self.Wc) + self.bc)

        c_next = f * c_prev + u * c_hat

        o = sigmoid(np.dot(concat_input, self.Wo) + self.bo)

        h_next = o * np.tanh(c_next)

        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
