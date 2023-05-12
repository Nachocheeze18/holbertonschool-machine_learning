#!/usr/bin/env python3
"""function to add 2 arrays"""


def add_arrays(arr1, arr2):\
    """adding arrays"""
if len(arr1) != len(arr2):
        return None
result = []
for i in range(len(arr1)):
    result.append(arr1[i] + arr2[i])
    return result
