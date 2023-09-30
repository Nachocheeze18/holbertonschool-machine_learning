#!/usr/bin/env python3
"""Imports"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model"""
    T = len(Observation)
    N, M = Emission.shape

    if Transition.shape != (N, N) or Initial.shape != (N, 1) or T <= 0:
        return None, None

    F = np.zeros((N, T))

    for i in range(N):
        F[i, 0] = Initial[i] * Emission[i, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[i, t-1] * Transition[i, j] * Emission[j, Observation[t]] for i in range(N))

    P = np.sum(F[i, T-1] for i in range(N))

    return P, F
