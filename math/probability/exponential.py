#!/usr/bin/env python3
"""Exponential"""

e = 2.7182818285
pi = 3.1415946536


class Exponential:
        """Eponential Class"""      
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1. / (sum(data) / float(len(data)))
