#!/usr/bin/env python3
"""RMSProp"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """RMS func"""
    n = beta2 * s + (1 - beta2) * np.power(grad, 2)
    r = var - alpha * grad / (np.sqrt(n) + epsilon)
    return r, n
