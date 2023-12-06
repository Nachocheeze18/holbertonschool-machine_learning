#!/usr/bin/env python3
"""Imports"""
import numpy as np


def uni_bleu(references, sentence):
    """Convert references to the required format"""
    ref_list = [[ref] for ref in references]

    max_len = max(len(ref) for ref in ref_list)
    ref_list = [ref + [None] * (max_len - len(ref)) for ref in ref_list]
    sentence = sentence + [None] * (max_len - len(sentence))

    precision = 0
    for w in sentence:
        match_counts = [
            np.count_nonzero(np.array(ref) == w) for ref in ref_list
        ]
        precision += np.max(match_counts) / len(sentence)

    bp = 1 if len(sentence) > len(ref_list[0][0]) else np.exp(1 - len(ref_list[0][0]) / len(sentence))
    
    score = bp * np.exp(np.log(precision))

    return score
