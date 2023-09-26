#!/usr/bin/env python3
"""Imports"""
import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset"""
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    gmm_model.fit(X)

    weight = gmm_model.weights_
    mean = gmm_model.means_
    cov = gmm_model.covariances_
    label = gmm_model.predict(X)

    bic = gmm_model.bic(X)

    return weight, mean, cov, label, bic
