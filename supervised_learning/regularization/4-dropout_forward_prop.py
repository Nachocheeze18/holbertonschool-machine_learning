#!/usr/bin/env python3
"""dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """forward prop using dropout"""
    cache = {'A0': X}
    A = X

    for l in range(1, L):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        Z = np.dot(W, A) + b
        A = np.tanh(Z)
        D = np.random.rand(*A.shape) < keep_prob
        A *= D / keep_prob
        cache['D' + str(l)] = D
        cache['A' + str(l)] = A

    W = weights['W' + str(L)]
    b = weights['b' + str(L)]
    Z = np.dot(W, A) + b
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    cache['A' + str(L)] = A

    return cache
