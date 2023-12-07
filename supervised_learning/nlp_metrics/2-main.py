#!/usr/bin/env python3

import numpy as np
ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["hello", "there", "my", "friend"]]
sentence = ["hello", "there", "comrade"]

print(np.round(ngram_bleu(references, sentence, 2), 10))