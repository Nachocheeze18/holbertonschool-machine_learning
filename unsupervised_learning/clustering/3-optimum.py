#!/usr/bin/env python3
"""Imports"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    results = []
    vars = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        cen, cluster = kmeans(X, k, iterations)
        results.append((cen, cluster))
        value = variance(X, cen)
        vars.append(value)

    ref = vars[0]
    d_vars = [ref - var for var in vars]

    return results, d_vars
