#!/usr/bin/env python3
"""Imports"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence"""
    reference_lengths = np.array([len(ref) for ref in references])
    sentence_length = len(sentence)

    # Calculate precision for each n-gram
    precisions = []
    for i in range(1, n + 1):
        reference_ngrams = np.concatenate([np.array([tuple(ref[j:j + i]) for j in range(len(ref) - i + 1)]) for ref in references])
        sentence_ngrams = np.array([tuple(sentence[j:j + i]) for j in range(sentence_length - i + 1)])

        # Calculate the number of overlapping n-grams
        overlaps = np.minimum(
            np.histogram(reference_ngrams, bins=np.unique(np.concatenate([reference_ngrams, sentence_ngrams])))[0],
            np.histogram(sentence_ngrams, bins=np.unique(np.concatenate([reference_ngrams, sentence_ngrams])))[0]
        )

        # Calculate precision for each n-gram
        precision = overlaps.sum() / len(sentence_ngrams)
        precisions.append(precision)

    # Calculate the geometric mean of precisions
    geo_mean = np.exp(np.mean(np.log(precisions)))

    # Calculate brevity penalty
    brevity_penalty = np.exp(1 - reference_lengths / sentence_length) if sentence_length < np.min(reference_lengths) else 1.0

    # Calculate BLEU score
    bleu_score = brevity_penalty * geo_mean

    return bleu_score