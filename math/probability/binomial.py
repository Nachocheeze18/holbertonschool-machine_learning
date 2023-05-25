#!/usr/bin/env python3
"""Binomial"""

e = 2.7182818285
pi = 3.1415946536

class Binomial:
    """binomial class"""
    def __init__(self, data=None, n=1, p=0.5):
        """constructer"""
                if data is None:
                if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Estimate n and p from the data
            self.n = len(data)
            successes = sum(data)
            self.p = float(successes) / self.n
