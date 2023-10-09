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
        """optimizes the black-box function"""
        X_opt = None
        Y_opt = None

        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if any(np.all(np.isclose(X_next, x)) for x in self.X):
                break

            Y_next = self.f(X_next)

            self.X = np.vstack((self.X, X_next))
            self.Y = np.vstack((self.Y, Y_next))

            self.gp.update(X_next, Y_next)

            if Y_opt is None or (self.minimize and Y_next < Y_opt):
                X_opt = X_next
                Y_opt = Y_next

            if Y_opt is None or (not self.minimize and Y_next > Y_opt):
                X_opt = X_next
                Y_opt = Y_next

        return X_opt, Y_opt