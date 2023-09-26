#!/usr/bin/env python3
"""Imports"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on the given"""
    linkage_matrix = scipy.cluster.hierarchy.linkage(
        X, method='ward')

    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix, color_threshold=dist, above_threshold_color='b')
    plt.show()

    labels = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, dist, criterion='distance')

    return labels
