#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering


def question_answer(question, reference):
    """snippet of text within a reference document to answer a question"""
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    input = tokenizer(question, reference, return_tensors="tf")

    output = model(input)

    start_index = tf.argmax(output.start_logits[0]).numpy().item()
    end_index = tf.argmax(output.end_logits[0]).numpy().item() + 1

    input_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"].numpy()[0])

    answer_tokens = input_tokens[start_index:end_index]

    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer.strip() if answer.strip() else None
