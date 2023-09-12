#!/usr/bin/env python3
"""Imports"""
import numpy as np


def intersection(x, n, P, Pr):
    """calculates the intersection of obtaining certain data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not all(0 <= p <= 1 for p in P) or not all(0 <= pr <= 1 for pr in Pr):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    likelihood = np.power(P, x) * np.power(1 - P, n - x)
    intersection = likelihood * Pr

    return intersection
