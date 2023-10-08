#!/usr/bin/env python3
"""Imports"""
import numpy as np


class GaussianProcess:
    """Guassian process class"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """constructor func"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Represents a noiseless 1D Gaussian process"""
        dist_matrix = np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)
        K = self.sigma_f**2 * np.exp(-0.5 * (dist_matrix / self.l)**2)
        return K
