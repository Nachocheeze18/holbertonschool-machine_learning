#!/usr/bin/env python3
"""Imports"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """embedding matrix"""
    if vocab==None:
        vec = CountVectorizer()
        X = vec.fit_transform(sentences)
        features = vec.get_feature_names_out()
    else:
        features = vocab
        vec = CountVectorizer(vocabulary=vocab)
        X = vec.transform(sentences)

    embeddings = X.toarray()

    return embeddings, features