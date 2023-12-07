#!/usr/bin/env python3
"""Imports"""
import numpy as np


def uni_bleu(references, sentence):
    """Convert references to the required format"""
    precisions = []
    for reference in references:
        reference_array = np.array(reference)
        sentence_array = np.array(sentence)
        common = np.intersect1d(reference_array, sentence_array)
        precision = len(common) / len(sentence)
        precisions.append(precision)

    closest_ref_len = min((len(ref)
                           for ref in references),
                           key=lambda ref_len: (abs(ref_len - len(sentence)), ref_len))
    brevity_penalty = np.exp(1 - closest_ref_len / len(sentence)) if len(
        sentence) < closest_ref_len else 1.0

    bleu = brevity_penalty * np.exp(np.mean(np.log(precisions)))

    return bleu
