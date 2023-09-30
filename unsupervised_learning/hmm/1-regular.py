#!/usr/bin/env python3
"""Imports"""
import numpy as np


def regular(P, tol=1e-8, max_iter=1000):
    """Determine the steady-state probabilities
    of a regular Markov chain."""

    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]

    pi = np.random.rand(1, n)
    pi /= np.sum(pi)

    for _ in range(max_iter):

        results = np.dot(pi, P)

        if np.linalg.norm(results - pi) < tol:
            return results

        pi = results

    return None
