#!/usr/bin/env python3
"""Imports"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding"""
    if vocab == None:
        vocab = set(word for sentence in sentences
                    for word in sentence.split())

    vec = TfidfVectorizer(vocabulary=vocab)

    embeddings = vec.fit_transform(sentences).toarray()

    features = vec.get_feature_names_out()

    z = embeddings, features
    return z
