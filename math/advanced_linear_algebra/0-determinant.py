#!/usr/bin/env python3
"""Matrix Determinant Calculator"""

def determinant(matrix):
    """
    Calculate the determinant of a square matrix.
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 0:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    result = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        cofactor = matrix[0][j] * determinant(sub_matrix)
        result += cofactor if j % 2 == 0 else -cofactor

    return result
