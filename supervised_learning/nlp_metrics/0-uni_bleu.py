#!/usr/bin/env python3
"""Imports"""
import numpy as np


def uni_bleu(references, sentence):
    """Convert references to the required format"""
    total_word_count_refs = min(len(ref) for ref in references)
    total_matches = sum(min(ref.count(word) for ref in references) for word in sentence)
    precision = total_matches / len(sentence) if len(sentence) > 0 else 0

    brevity_penalty = np.exp(1 - total_word_count_refs / len(sentence)) if len(sentence) < total_word_count_refs else 1

    bleu_score = brevity_penalty * precision

    return bleu_score
