#!/usr/bin/env python3
"""Imports"""
from math import exp, log
from collections import Counter

def ngram_bleu(references, sentence, n):
    """
    Calculate the n-gram BLEU score for a sentence.

    Args:
    references (list): List of reference translations.
    sentence (list): List containing the model proposed sentence.
    n (int): Size of the n-gram to use for evaluation.

    Returns:
    float: The n-gram BLEU score.
    """

    def calculate_precision(candidate, reference, n):
        candidate_ngrams = [tuple(candidate[i:i + n]) for i in range(len(candidate) - n + 1)]
        reference_ngrams = [tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)]

        candidate_ngram_counts = Counter(candidate_ngrams)
        reference_ngram_counts = Counter(reference_ngrams)

        clipped_counts = {ngram: min(candidate_ngram_counts[ngram], reference_ngram_counts[ngram]) for ngram in candidate_ngram_counts}

        total_clipped_counts = sum(clipped_counts.values())
        total_candidate_counts = sum(candidate_ngram_counts.values())

        precision = total_clipped_counts / total_candidate_counts if total_candidate_counts > 0 else 0.0

        return precision

    def brevity_penalty(candidate, references):
        candidate_length = len(candidate)
        reference_lengths = [len(reference) for reference in references]
        closest_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
        brevity_penalty = 1.0 if candidate_length >= closest_length else exp(1 - closest_length / candidate_length)

        return brevity_penalty

    precision_scores = []

    for i in range(n):
        precision_i = sum(calculate_precision(sentence, reference, i + 1) for reference in references) / len(references)
        precision_scores.append(precision_i)

    geometric_mean = (precision_i ** (1 / n) for precision_i in precision_scores)
    bleu_score = brevity_penalty(sentence, references) * exp(sum(log(precision_i) for precision_i in geometric_mean) / n)

    return bleu_score