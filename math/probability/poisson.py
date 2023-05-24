#!/usr/bin/env python3
"""Poisson"""


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
        """convert k to an interger if not already"""
        k = int(k)
        
        """Check if k is out of range"""
        if k < 0 or k >= len(self.data):
            return 0
        
        """Calculate the PMF value"""
        total_count = sum(self.data)
        pmf_value = self.data[k] / total_count
        
        return pmf_value
    