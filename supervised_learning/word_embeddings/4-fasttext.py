#!/usr/bin/env python3
"""Imports"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5,
                   negative=5, window=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """Create and train a Gensim FastText model."""
    skip_gram = 0 if cbow else 1

    model = FastText(
        sentences, vector_size=size, min_count=min_count,
        window=window, negative=negative, sg=skip_gram,
        epochs=iterations, seed=seed, workers=workers
    )
    
    return model