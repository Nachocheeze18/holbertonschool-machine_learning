#!/usr/bin/env python3
"""stop descent"""

def early_stopping(cost, opt_cost, threshold, patience, count):
    """early stopping conditions"""
    if cost - opt_cost > threshold:
        count += 1
    else:
        count = 0

    stop = False
    if count >= patience:
        stop = True

    return stop, count
