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

    visited = set()
    stack = [0]

    while stack:
        state = stack.pop()
        visited.add(state)
        for next_state in range(n):
            if P[state, next_state] > 0 and next_state not in visited:
                stack.append(next_state)

    if len(visited) != n:
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
