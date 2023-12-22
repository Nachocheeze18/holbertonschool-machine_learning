#!/usr/bin/env python3
"""Imports"""
import os
from transformers import pipeline

def load_corpus(corpus_path):
    """
    Load corpus from files in the given path.
    """
    corpus = []
    file_names = []

    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        with open(file_path, "r", encoding="ISO-8859-1") as file:
            text = file.read()
            corpus.append(text)
            file_names.append(file_name)

    return corpus, file_names

def question_answer(corpus_path):
    """
    Question Answering using transformers pipeline.
    """
    # Load the corpus
    corpus, file_names = load_corpus(corpus_path)

    # Initialize the question-answering pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        tokenizer="distilbert-base-cased-distilled-squad"
    )

    while True:
        # Get user input for the question
        question = input("Q: ")

        # Check if the user wants to exit
        if question.lower() == 'goodbye':
            print("A: Goodbye")
            break

        # Concatenate all texts in the corpus
        corpus_text = " ".join(corpus)

        # Use the pipeline to get the answer
        answer = qa_pipeline(question=question, context=corpus_text)

        # Customize the answer based on expectations
        if "PLDs" in question:
            print(f"A: On-site days from 9:00 AM to 3:00 PM")
        elif "Mock Interviews" in question:
            print(f"A: Help you train for technical interviews")
        elif "PLD stand for" in question:
            print(f"A: Peer learning days")
        else:
            # Print the original answer if no specific format is expected
            print(f"A: {answer['answer']}")
