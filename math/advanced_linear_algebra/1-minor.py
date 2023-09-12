#!/usr/bin/env python3
"""Minor Matrix"""


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)

    if num_rows == 0:
        return 1

    if num_rows == 1:
        return matrix[0][0]

    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(num_rows):
        submatrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = matrix[0][col] * determinant(submatrix)
        if col % 2 == 0:
            det += cofactor
        else:
            det -= cofactor

    return det


def minor(matrix):
    """calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = []

    for i in range(num_rows):
        det = []
        for j in range(num_rows):
            submatrix = [row[:j] + row[j + 1:] for row in
                         (matrix[:i] + matrix[i + 1:])]
            det.append(determinant(submatrix))
        minor_matrix.append(det)

    return minor_matrix
