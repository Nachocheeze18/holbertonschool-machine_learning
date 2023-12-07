#!/usr/bin/env python3
"""Imports"""
import numpy as np


def cumulative_bleu(references, sentence, N):
    """Returns: the cumulative bleu score"""
    def make_ngrams(sentence, n):
        """converts a sentence to ngrams"""
        ngrams = [' '.join(sentence[i:i+n]) for i in range(len(
            sentence) - n + 1)]
        return ngrams

    def count_ngrams(sentence, ngrams):
        ngram_count = {ngram: 0 for ngram in ngrams}
        for ngram in make_ngrams(sentence, len(ngrams[0].split())):
            if ngram in ngrams:
                ngram_count[ngram] += 1

        return ngram_count

    Pn = []
    for n in range(1, N + 1):
        ngrams = make_ngrams(sentence, n)

        c = len(sentence)
        r = min(len(reference) for reference in references)
        BP = 1 if c > r else np.exp(1 - (r / c))

        max_ref_count = {ngram: 0 for ngram in ngrams}
        for reference in references:
            ngram_count = count_ngrams(reference, ngrams)
            for ngram in ngrams:
                max_ref_count[ngram] = max(
                    ngram_count[ngram], max_ref_count[ngram])

        precision = np.sum(list(max_ref_count.values())) / np.sum(
            list(count_ngrams(sentence, ngrams).values()))
        Pn.append(precision)

    if 0 in Pn:
        Pn.remove(0)

    bleu = BP * np.exp(np.sum((1/N) * np.log(Pn)))

    return bleu
