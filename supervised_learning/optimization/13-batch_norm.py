#!/usr/bin/env python3
"""neural network"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """batch normalization"""
    m = np.mean(Z, axis=0)
    r = np.var(Z, axis=0)
    n = (Z - m) / np.sqrt(r + epsilon)
    s = gamma * n + beta
    return s
