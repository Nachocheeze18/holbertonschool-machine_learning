#!/usr/bin/env python3
"""Imports"""
import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, k):
    """performs K-means on a dataset"""
    n, d = X.shape

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(X)

    C = kmeans.cluster_centers_

    clss = kmeans.labels_
    
    return C, clss
