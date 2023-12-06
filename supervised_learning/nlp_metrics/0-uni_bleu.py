#!/usr/bin/env python3
"""Imports"""
import numpy as np


def uni_bleu(references, sentence):
    """Convert references to the required format"""
    total_word_count_refs = max(len(ref) for ref in references)
    total_matches = sum(min(ref.count(word) for ref in references) for word in sentence)
    precision = total_matches / len(sentence) if len(sentence) > 0 else 0

    perfect_match = all(word in ref for word in sentence for ref in references)

    brevity_penalty = 1 if perfect_match else np.exp(1 - total_word_count_refs / len(sentence))

    bleu_score = brevity_penalty * precision

    return bleu_score
