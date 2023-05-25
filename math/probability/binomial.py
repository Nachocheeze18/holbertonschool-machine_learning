#!/usr/bin/env python3
"""Binomial"""

e = 2.7182818285
pi = 3.1415946536

class Binomial:
    """binomial class"""
    def __init__(self, data=None, n=1, p=0.5):
        """constructer"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            # Estimate n and p from the data
            self.n = len(data)
            successes = sum(data)
            self.p = float(successes) / self.n

            mean = sum(data) / len(data)
            v = 0
            for number in data:
                v = v + (number - mean) ** 2
            v = v / len(data)
            q = v / mean
            p1 = 1 - q
            n1 = (sum(data) / p1) / len(data)
            self.n = int(round(n1))
            self.p = float(mean/self.n)

    def pmf(self, k):
        """pmf calculations"""

        k = int(k)

        if k < 0:
            return 0

        c = (self.factorial(k) * self.factorial(self.n - k))
        a = (self.factorial(self.n) / c)

        return (a * (self.p ** k) * ((1 - self.p) ** (self.n - k)))

    def factorial(self, k):
        """factorial helper function"""
        result = 1
        for i in range(1, k+1):
            result *= i
        return result

    def cdf(self, k):
        """cdf calculation"""
        if k < 0:
            return 0

        i = int(k)
        prob = 0

        for x in range(i + 1):
            prob += self.pmf(x)

        return (prob)
    