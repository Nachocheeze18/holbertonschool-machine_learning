#!/usr/bin/env python3
"""Imports"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian class"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """performs Bayesian optimization on a noiseless 1D Gaussian process"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.X = X_init  # Initialize self.X with X_init
        self.Y = Y_init  # Initialize self.Y with Y_init
    
    def acquisition(self):
        """calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        op = self.X_s[np.argmax(ei)] if self.minimize else self.X_s[
            np.argmin(ei)]
        return op, ei

    def optimize(self, iterations=100):
        for _ in range(iterations):
            x_next, _ = self.acquisition()

            # Check if the proposed point has already been sampled
            if any(np.all(np.isclose(x_next, x)) for x in self.X):
                break

            y_next = self.f(x_next)
            self.X = np.vstack((self.X, x_next))
            self.Y = np.vstack((self.Y, y_next))

        if self.minimize:
            x_opt_idx = np.argmin(self.Y)
        else:
            x_opt_idx = np.argmax(self.Y)

        x_opt = self.X[x_opt_idx]
        y_opt = self.Y[x_opt_idx]

        return x_opt, y_opt