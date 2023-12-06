#!/usr/bin/env python3
"""Imports"""
from keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    """Convert a Gensim Word2Vec model to a Keras Embedding layer."""
    vocab_size, embedding_dim = model.wv.vectors.shape

    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for i, word in enumerate(model.wv.index_to_key):
        embedding_vector = model.wv[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    )
    
    return embedding_layer
