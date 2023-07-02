#!/usr/bin/env python3
"""stop descent"""

def early_stopping(cost, opt_cost, threshold, patience, count):
    """early stopping conditions"""
    if isinstance(cost - opt_cost, bool):
        count = 0
    else:
        count += 1

    return count >= patience, count
