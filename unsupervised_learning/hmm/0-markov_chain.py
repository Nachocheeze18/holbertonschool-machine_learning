#!/usr/bin/env python3
"""Imports"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain
    being in a particular state after a specified number of iterations
    """
    if (
        P.shape[0] != P.shape[1]
        or s.shape[0] != 1
        or P.shape[0] != s.shape[1]
        or t < 1
    ):
        return None

    n = P.shape[0]
    result = s.copy()

    for _ in range(t):
        result = np.dot(result, P)

    return result
