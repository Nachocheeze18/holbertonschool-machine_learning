# Natural Language Processing (NLP)

NLP is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The goal is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant.
## Word Embedding:

Word embedding is a technique in NLP that represents words as vectors in a continuous vector space. This representation captures semantic relationships between words, and it is useful for various NLP tasks like sentiment analysis, language translation, and information retrieval.
## Bag of Words (BoW):

BoW is a simple and common technique for text representation in NLP. It involves representing a document as an unordered set of words, disregarding grammar and word order but keeping track of word frequency. This results in a "bag" of words.
## TF-IDF (Term Frequency-Inverse Document Frequency):

TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. It considers both the frequency of a term in a document (TF) and its inverse document frequency (IDF), which measures how unique a term is across the entire document collection.
## CBOW (Continuous Bag of Words):

CBOW is a type of neural network architecture used for training word embeddings. It predicts the target word based on the context of surrounding words.
## Skip-Gram:

Skip-gram is another neural network architecture for training word embeddings. It does the opposite of CBOW; it predicts the context words based on a given target word.
## N-gram:

An n-gram is a contiguous sequence of n items (words or characters) from a given sample of text or speech. For example, a bigram (2-gram) would be a two-word sequence.
## Negative Sampling:

Negative sampling is a technique used during the training of word embeddings to improve efficiency. Instead of adjusting all weights for each training example, negative sampling focuses on a small, randomly chosen set of "negative" examples.
## Word2Vec, GloVe, fastText, ELMo:

These are different algorithms and models for generating word embeddings.
Word2Vec: Developed by Google, it uses CBOW and Skip-gram models to learn word embeddings.
GloVe (Global Vectors for Word Representation): It is based on global statistics of the corpus and creates a direct representation of the words in vector space.
fastText: Developed by Facebook, it is an extension of Word2Vec that considers sub-word information.
ELMo (Embeddings from Language Models): It generates word representations by considering the context and producing contextualized embeddings.