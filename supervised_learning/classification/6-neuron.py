#!/usr/bin/env python3
"""nueron"""
import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Cunstructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """Forward Propogation"""
        self.__A = self.sigmoid(np.matmul(self.__W, X) + self.__b)
        return self.__A

    def sigmoid(self, X):
        """Sigmoid"""
        return 1 / (1 + np.exp(-X))

    def cost(self, Y, A):
        """model cost"""
        m = Y.shape[1]
        cs = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        """cs = cost"""
        return cs

    def evaluate(self, X, Y):
        """Evaluate func"""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient Descent"""
        m = X.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.mean(dZ)

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train func"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
