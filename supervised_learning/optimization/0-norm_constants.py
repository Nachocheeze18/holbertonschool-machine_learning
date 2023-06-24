#!/usr/bin/env python3
"""calculates normilization"""

import numpy as np


def normalization_constants(X):
    """normal func"""
    mn = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    return mn, std