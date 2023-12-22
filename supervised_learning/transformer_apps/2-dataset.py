#!/usr/bin/env python3
"""Imports"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

class Dataset:
    """Class to handle translation datasets."""
    def __init__(self):
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Initialize the Dataset."""
        all_text_pt = []
        all_text_en = []
        for pt, en in data:
            all_text_pt.append(pt.numpy())
            all_text_en.append(en.numpy())

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            all_text_pt, target_vocab_size=2**15
        )

        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            all_text_en, target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a translation into tokens."""
        pt_t = self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_t = self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_t
        en_tokens = [self.tokenizer_en.vocab_size] + en_t
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
    print('got here')
    for pt, en in data.data_train.take(1):
        print(pt, en)
    for pt, en in data.data_valid.take(1):
        print(pt, en)