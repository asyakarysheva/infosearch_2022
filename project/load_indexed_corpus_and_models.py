"""
В этом модуле загружаются индексированные разными способами корпуса и векторизаторы.
Также загружаются токенизатор и модель BERT, необходимые для векторизации запроса.
"""

import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from paths import index_TfIdf_path, vectorizer_TfIdf_path, index_BM_path, vectorizer_BM_path, index_BERT_path


def load_TfIdf(index_TfIdf_path, vectorizer_TfIdf_path):  # загрузка индексированного корпуса и
    with open(index_TfIdf_path, 'rb') as file:
        index_TfIdf = pickle.load(file)
    with open(vectorizer_TfIdf_path, 'rb') as file:
        vectorizer_TfIdf = pickle.load(file)
    return index_TfIdf, vectorizer_TfIdf


def load_BM25(index_BM_path, vectorizer_BM_path):
    with open(index_BM_path, 'rb') as file:
        index_BM = pickle.load(file)
    with open(vectorizer_BM_path, 'rb') as file:
        vectorizer_BM = pickle.load(file)
    return index_BM, vectorizer_BM


def load_BERT_corpus(index_BERT_path):
    answers_embeddings = torch.load(index_BERT_path, map_location=torch.device('cpu'))
    answers_embeddings_array = []
    for item in answers_embeddings:
        item = item.detach().numpy()  # элементы списка приводятся к классу numpy.ndarray
        answers_embeddings_array.append(item)
    # Далее создаётся матрица типа numpy.ndarray:
    # Её размер – (50000, 1024)
    index_BERT = np.array(answers_embeddings_array)
    return index_BERT


def load_tokenizer_BERT():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return tokenizer


def load_model_BERT():
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return model


index_TfIdf, vectorizer_TfIdf = load_TfIdf(index_TfIdf_path, vectorizer_TfIdf_path)
index_BM, vectorizer_BM = load_BM25(index_BM_path, vectorizer_BM_path)
index_BERT = load_BERT_corpus(index_BERT_path)
tokenizer_BERT = load_tokenizer_BERT()
model_BERT = load_model_BERT()
