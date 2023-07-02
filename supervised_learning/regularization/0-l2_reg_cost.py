#!/usr/bin/env python3
"""neural network"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """L2 regularization"""
    regularization = 0.0

    for l in range(1, L + 1):
        W = weights["W" + str(l)]
        regularization += np.sum(np.square(W))

    regularization *= (lambtha / (2 * m))
    total_cost = cost + regularization

    return total_cost
