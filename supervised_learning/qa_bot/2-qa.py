#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering


def answer_loop(reference):
   """answers questions from a reference text:"""
   tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
   model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

   exit_commands = ["exit", "quit", "goodbye", "bye"]

   while True:
       user_question = input("Q: ").strip().lower()

       if user_question in exit_commands:
           print("A: Goodbye")
           break

       if not user_question:
           print("A: Please enter a valid question.")
           continue

       input_text = f"Question: {user_question} Context: {reference}"

       try:
           encoded_input = tokenizer(input_text, return_tensors="tf")

           outputs = model(encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])

           start_logits, end_logits = outputs.start_logits.numpy(), outputs.end_logits.numpy()

           start_index = min(tf.argmax(start_logits, axis=1).numpy()[0], len(encoded_input["input_ids"].numpy()[0]) - 1)
           end_index = min(tf.argmax(end_logits, axis=1).numpy()[0] + 1, len(encoded_input["input_ids"].numpy()[0]))

           answer_tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"].numpy()[0][start_index:end_index])

           answer = tokenizer.convert_tokens_to_string(answer_tokens)

           if answer.strip():
               print("A:", answer)
           else:
               print("A: Sorry, I do not understand your question.")

       except Exception as e:
           print(f"A: An error occurred: {str(e)}")
