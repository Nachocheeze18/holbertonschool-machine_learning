#!/usr/bin/env python3
"""Poisson"""


e = 2.7182818285
pi = 3.1415926536


class Poisson:
    """Poisson Class"""
    def __init__(self, data=None, lambtha=1.):
        """Constructs the value"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """pmf function"""

        def factorial(n):
            """factorial function"""

            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

        k = int(k)
        if k < 0:
            return 0

        top = e ** (-self.lambtha) * self.lambtha ** (k)
        bot = factorial(k)
        sum = top / bot
        return sum

    def cdf(self, k):
         """cdf function"""
        k = int(k)
        if k < 0:
            return 0

        cdf_sum = 0
        for i in range(k + 1):
            cdf_sum += self.pmf(i)

        return cdf_sum