#!/usr/bin/env python3
"""dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """forward prop using dropout"""
    cache = {}
    A = X

    for l in range(1, L):
        W = weights["W" + str(l)]
        b = weights["b" + str(l)]
        Z = np.dot(W, A) + b
        A = np.tanh(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A *= D
        A /= keep_prob
        cache["Z" + str(l)] = Z
        cache["A" + str(l)] = A
        cache["D" + str(l)] = D

    W = weights["W" + str(L)]
    b = weights["b" + str(L)]
    Z = np.dot(W, A) + b
    exp_Z = np.exp(Z)
    A = exp_Z / np.sum(exp_Z, axis=0)
    cache["Z" + str(L)] = Z
    cache["A" + str(L)] = A

    return cache
