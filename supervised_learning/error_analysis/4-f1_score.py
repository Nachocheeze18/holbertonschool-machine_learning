#!/usr/bin/env python3
"""F1 score"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the true positives, false positives, and false negatives"""
    n = precision(confusion) * sensitivity(confusion)
    d = precision(confusion) + sensitivity(confusion)
    f1 = 2 * (n / d)
    return f1