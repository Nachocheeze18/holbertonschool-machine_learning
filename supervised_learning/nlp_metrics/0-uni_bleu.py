#!/usr/bin/env python3
"""Imports"""
import numpy as np


def uni_bleu(references, sentence):
    """Convert references to the required format"""
    word_counts = {word: min([ref.count(word) for ref in references], default=0) for word in sentence}
    total_matches = sum(word_counts.values())
    precision = total_matches / len(sentence) if len(sentence) > 0 else 0

    reference_length = min(len(ref) for ref in references)
    brevity_penalty = np.exp(1 - (reference_length / len(sentence))) if len(sentence) < reference_length else 1

    brevity_penalty = 1

    bleu_score = brevity_penalty * precision

    return bleu_score
