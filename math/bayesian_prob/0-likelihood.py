#!/usr/bin/env python3
"""Import"""
import numpy as np


def likelihood(x, n, P):
    """calculates the likelihood of obtaining a specific data outcome"""
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) != int or x < 0:
        raise ValueError
    ('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.shape != (len(P),):
        raise TypeError('P must be a 1D numpy.ndarray')
    if any(val < 0 or val > 1 for val in P):
        raise ValueError('All values in P must be in the range [0, 1]')

    fact = np.math.factorial
    prob = [(fact(n) / (fact(x) * fact(n - x)))
            * (p ** x) * ((1 - p) ** (n - x)) for p in P]
    return np.array(prob)
