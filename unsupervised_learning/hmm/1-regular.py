#!/usr/bin/env python3
"""Imports"""
import numpy as np


def regular(P):
    """Determine the steady-state probabilities
    of a regular Markov chain."""
    n, m = P.shape

    # Check if P is a square matrix
    if n != m:
        return None

    # Check if P is a regular matrix (all rows sum to 1)
    if not np.allclose(np.sum(P, axis=1), 1.0):
        return None

    # Compute the steady state probabilities
    A = np.transpose(P) - np.identity(n)
    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1

    try:
        pi = np.linalg.solve(A, b)
        return pi
    except np.linalg.LinAlgError:
        return None
