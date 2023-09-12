#!/usr/bin/env python3
"""Minor Matrix"""


def minor(matrix):
    """calculates the minor marix of a matrix"""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = [[0 for _ in range(num_rows)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_rows):
            submatrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            det = determinant(submatrix)
            minor_matrix[i][j] = det

    return minor_matrix

def determinant(matrix):
    """
    Calculate the determinant of a square matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(row):
            raise ValueError("matrix must be a square matrix")

    num_rows = len(matrix)

    if num_rows == 0:
        return 1

    if num_rows == 1:
        return matrix[0][0]

    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(num_rows):
        submatrix = [[matrix[i][j] for j in range(num_rows) if j != col] for i in range(1, num_rows)]
        cofactor = matrix[0][col] * determinant(submatrix)
        if col % 2 == 0:
            det += cofactor
        else:
            det -= cofactor

    return det
