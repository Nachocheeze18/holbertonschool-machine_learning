#!/usr/bin/env python3
"""Imports"""
import numpy as np


def uni_bleu(references, sentence):
    """Convert references to the required format"""
    ref_list = [[ref] for ref in references]

    max_len = max(len(ref) for ref in ref_list)
    ref_list = [ref + [None] * (
        max_len - len(ref)) for ref in ref_list]
    sentence = sentence + [None] * (
        max_len - len(sentence))

    match_counts = 0
    total_counts = 0
    for w in sentence:
        if w is not None:
            match_counts += max(
                [ref.count(w) for ref in ref_list])
            total_counts += 1

    precision = match_counts / total_counts

    bp = min(1, len(sentence) / len(ref_list[0][0]))

    score = bp * np.exp(np.log(precision))

    return score
