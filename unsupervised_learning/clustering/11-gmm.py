#!/usr/bin/env python3
"""Imports"""
import sklearn.mixture

def gmm(X, k):
    """calculates a GMM from a dataset"""
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    gmm_model.fit(X)

    weights, means, covariances = gmm_model.weights_,
    gmm_model.means_, gmm_model.covariances_
    
    labels = gmm_model.predict(X)

    bic = gmm_model.bic(X)

    return weights, means, covariances, labels, bic
