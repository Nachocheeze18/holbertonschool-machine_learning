#!/usr/bin/env python3
"""dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """forward prop using dropout"""
    cache = {'A0': X}

    for layer in range(1, L + 1):
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]
        Z = np.matmul(W, cache["A{}".format(layer - 1)]) + b

        if layer == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A *= D / keep_prob
            cache["D{}".format(layer)] = D

        cache["A{}".format(layer)] = A

    return cache
