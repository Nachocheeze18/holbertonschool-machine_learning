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

    Q = P[:-1, :-1]

    R = P[:-1, -1]

    I = np.identity(n-1)
    F = np.ones((n-1,))

    try:
        inv = np.linalg.inv(I - Q)
        pi = np.dot(inv, F)
        steady_state = np.append(pi, 1 - np.sum(pi))
        return steady_state
    except np.linalg.LinAlgError:
        return None
