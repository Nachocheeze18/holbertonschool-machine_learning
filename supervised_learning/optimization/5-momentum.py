#!/usr/bin/env python3
"""variable momentum"""

import numpy as np

def update_variables_momentum(alpha, beta1, var, grad, v):
    """momentum func"""
    n = beta1 * v + (1 - beta1) * grad
    r = var - alpha * new_v
    return r, n