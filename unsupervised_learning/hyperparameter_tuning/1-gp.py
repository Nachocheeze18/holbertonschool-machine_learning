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

    def predict(self, X_s):
        """predicts the mean and standard deviation of points in a Gaussian process"""
        Kernel_s = self.kernel(X_s, self.X)
        Kernel_ss = self.kernel(X_s, X_s)
        Kernel_inv = np.linalg.inv(self.K)

        mean = Kernel_s.dot(Kernel_inv).dot(self.Y)
        cov = Kernel_ss - Kernel_s.dot(Kernel_inv).dot(Kernel_s.T)
        mu = mean.squeeze()
        sigma = np.sqrt(np.diag(cov))

        return mu, sigma

    def kernel(self, X1, X2):
        """Represents a noiseless 1D Gaussian process"""
        dist_matrix = np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)
        K = self.sigma_f**2 * np.exp(-0.5 * (dist_matrix / self.l)**2)
        return K

