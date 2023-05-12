#!/usr/bin/env python3
"""matrix multiplication"""


def mat_mul(mat1, mat2):
    # Get the dimensions of the matrices
    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])

    # Check if the matrices can be multiplied
    if cols1 != rows2:
        return None

    # Create a new matrix to store the result
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    # Perform matrix multiplication
    for i in range(rows1):
        for j in range(cols2):
            for k in range(rows2):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
