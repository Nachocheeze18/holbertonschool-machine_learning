#!/usr/bin/env python3
"""Imports"""
import numpy as np


def marginal(x, n, P, Pr):
    """calculates the marginal probability of obtaining the data"""
    value = 'x must be an integer that is greater than or equal to 0'
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(value)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    fact_n = fact_x = fact_n_minus_x = 1
    for i in range(1, n + 1):
        fact_n *= i
    for i in range(1, x + 1):
        fact_x *= i
    for i in range(1, n - x + 1):
        fact_n_minus_x *= i

    comb = fact_n / (fact_x * fact_n_minus_x)

    likelihood = comb * np.power(P, x) * np.power(1 - P, n - x)

    inter = likelihood * Pr
    return np.sum(inter)
