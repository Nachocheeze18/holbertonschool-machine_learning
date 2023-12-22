import os
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    # Load the pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    # Load the corpus documents
    corpus = []
    filenames = []
    for filename in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            corpus.append(text)
            filenames.append(filename)

    # Tokenize and encode the sentence
    sentence_tokens = tokenizer(sentence, return_tensors='pt')
    sentence_embedding = model(**sentence_tokens).last_hidden_state.mean(dim=1)

    # Tokenize and encode the corpus documents
    corpus_tokens = tokenizer(corpus, return_tensors='pt', padding=True, truncation=True)
    corpus_embeddings = model(**corpus_tokens).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity between the sentence and each document
    similarities = cosine_similarity(sentence_embedding, corpus_embeddings)[0]

    # Find the index of the most similar document
    most_similar_index = similarities.argmax()

    # Return the reference text of the most similar document
    most_similar_text = corpus[most_similar_index]
    most_similar_filename = filenames[most_similar_index]

    return most_similar_filename, most_similar_text
