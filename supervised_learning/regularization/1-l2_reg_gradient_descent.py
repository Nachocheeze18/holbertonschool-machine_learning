#!/usr/bin/env python3
"""weight and biases"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """gradient descent with l2 regularzation"""
    m = Y.shape[1]

    dZ = cache["A" + str(L)] - Y

    for l in range(L, 0, -1):
        prev = cache["A" + str(l-1)]
        W = weights["W" + str(l)]
        b = weights["b" + str(l)]

        dW = (1 / m) * np.dot(dZ, prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        A_prev = np.dot(W.T, dZ)

        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db

        dZ = A_prev * (1 - np.power(prev, 2)) if l > 1 else A_prev

    return weights
