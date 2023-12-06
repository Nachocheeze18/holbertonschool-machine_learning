#!/usr/bin/env python3
"""Imports"""
import numpy as np


def uni_bleu(references, sentence):
    """Convert references to the required format"""
    word_counts = {word: min([ref.count(word) for ref in references], default=0) for word in sentence}
    total_counts = sum(word_counts.values())
    precision = total_counts / len(sentence) if len(sentence) > 0 else 0

    ref_lengths = [len(ref) for ref in references]
    shortest_ref_length = min(ref_lengths)

    brevity_penalty = np.exp(1 - (shortest_ref_length / len(sentence))) if len(sentence) < shortest_ref_length else 1

    bleu_score = brevity_penalty * precision

    return bleu_score
