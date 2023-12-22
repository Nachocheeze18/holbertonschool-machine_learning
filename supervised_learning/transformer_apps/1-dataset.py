#!/usr/bin/env python3
"""Imports"""
import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf

class Dataset:
    """Class to handle translation datasets."""

    def __init__(self):
        """Initialize the Dataset."""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self._build_tokenizers(self.data_train)

    def _build_tokenizers(self, data):
        """Build subword text encoders for Portuguese and English."""
        pt_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        en_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return pt_tokenizer, en_tokenizer

    def encode(self, pt, en):
        """Encode a translation into tokens."""
        pt_tokens = np.array([self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1])
        en_tokens = np.array([self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1])
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for the encode method."""
        pt_tokens, en_tokens = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens

if __name__ == "__main__":
    import tensorflow as tf

    data = Dataset()
    for pt, en in data.data_train.take(1):
        print(data.encode(pt, en))
    for pt, en in data.data_valid.take(1):
        print(data.encode(pt, en))