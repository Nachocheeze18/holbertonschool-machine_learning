#!/usr/bin/env python3
"""Imports"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """Create and train a Gensim Word2Vec model."""
    skip_gram = 0 if cbow else 1

    model = Word2Vec(
        sentences, vector_size=size, min_count=min_count,
                     window=window, negative=negative, sg=skip_gram,
                     epochs=iterations, seed=seed, workers=workers)

    return model
