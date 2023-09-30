#!/usr/bin/env python3
"""Imports"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray):
        return False

    n = P.shape[0]

    if P.shape != (n, n):
        return False

    absorbing_states = set()
    for i in range(n):
        if P[i, i] == 1:
            absorbing_states.add(i)

    if not absorbing_states:
        return False

    for i in absorbing_states:
        for j in range(n):
            if j not in absorbing_states and P[i, j] != 0:
                return False

    return True
