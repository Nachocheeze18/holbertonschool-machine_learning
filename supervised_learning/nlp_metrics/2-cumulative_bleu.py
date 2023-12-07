from collections import Counter

def cumulative_bleu(references, sentence, n):
    """Calculate cumulative n-gram BLEU score for a sentence."""

    def calculate_precision(candidate, references, n):
        """Calculate precision for a given n-gram."""
        candidate_ngrams = Counter(zip(*[candidate[i:] for i in range(n)]))
        mrn = Counter()

        for ref in references:
            reference_ngrams = Counter(zip(*[ref[i:] for i in range(n)]))
            mrn += reference_ngrams

        clipped_ngrams = {ngram: min(candidate_ngrams[ngram], mrn[ngram]) for ngram in candidate_ngrams}

        precision = sum(clipped_ngrams.values()) / max(1, sum(candidate_ngrams.values()))

        return precision

    bleu = 1.0

    for i in range(1, n + 1):
        precision_i = calculate_precision(sentence, references, i)
        bleu *= precision_i

    reference_length = min(len(ref) for ref in references)
    candidate_length = len(sentence)
    
    brevity_penalty = min(1, reference_length / candidate_length)

    bleu = brevity_penalty * bleu ** (1/n)

    return bleu