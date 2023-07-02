#!/usr/bin/env python3
"""dropout"""

import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """dropout gradient descent"""
    m = Y.shape[1]
    grads = {}

    dZ = cache['A' + str(L)] - Y
    grads['dW' + str(L)] = 1/m * np.dot(dZ, cache['A' + str(L-1)].T)
    grads['db' + str(L)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

    for l in reversed(range(1, L)):
        dA_prev = np.dot(weights['W' + str(l+1)].T, dZ)
        dA_prev = dA_prev * cache['D' + str(l)] / keep_prob
        dZ = dA_prev * (1 - np.power(cache['A' + str(l)], 2))
        grads['dW' + str(l)] = 1/m * np.dot(dZ, cache['A' + str(l-1)].T)
        grads['db' + str(l)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

    for l in range(1, L+1):
        weights['W' + str(l)] -= alpha * grads['dW' + str(l)]
        weights['b' + str(l)] -= alpha * grads['db' + str(l)]
