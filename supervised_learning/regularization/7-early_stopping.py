#!/usr/bin/env python3
"""stop descent"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """early stopping conditions"""
    count = 0 if (opt_cost - cost) > threshold else count + 1
    return (True, count) if count >= patience else (False, count)
