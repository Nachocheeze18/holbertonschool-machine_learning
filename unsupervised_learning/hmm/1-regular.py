#!/usr/bin/env python3
"""Imports"""
import numpy as np


def regular(P):
    """Determine the steady-state probabilities
    of a regular Markov chain."""
    n, m = P.shape

    if n != m:
        return None

    if not np.allclose(np.sum(P, axis=1), 1.0):
        return None

    A = np.transpose(P) - np.identity(n)
    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1

    try:
        pi = np.linalg.solve(A, b)
        return np.array([pi])
    except np.linalg.LinAlgError:
        return None
