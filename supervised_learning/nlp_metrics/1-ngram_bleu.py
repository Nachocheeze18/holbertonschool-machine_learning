#!/usr/bin/env python3
"""Imports"""
import numpy as np
from collections import Counter


def ngram_bleu(references, candidate_sentence, n):
    """BLEU score calculation."""
    candidate_ngrams = Counter(zip
                               (*[candidate_sentence[i:] for i in range(n)]))
    reference_ngrams_total = Counter()

    for reference in references:
        reference_ngrams = Counter(zip
                                   (*[reference[i:] for i in range(n)]))
        reference_ngrams_total += reference_ngrams

    clipped_counts = {ngram: min(candidate_ngrams[ngram],
                                 reference_ngrams_total[ngram]
                                 ) for ngram in candidate_ngrams}
    total_clipped_counts = sum(clipped_counts.values())
    total_candidate_counts = sum(candidate_ngrams.values())

    precision = total_clipped_counts / max(1, total_candidate_counts)

    closest_reference_length = min(len(reference) for reference in references)
    brevity_penalty = np.exp(1 - (closest_reference_length / len(
        candidate_sentence))) if len(
            candidate_sentence) < closest_reference_length else 1

    bleu_score = brevity_penalty * precision

    return bleu_score
