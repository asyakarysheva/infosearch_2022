"""
Этот модуль нужен для подсчёта близости между векторизованном запросом и индексированным корпусом.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def computing_similarity_TfIdf(index_TfIdf, vectorizer_TfIdf, query_preprocessed):
    index_query = vectorizer_TfIdf.transform([query_preprocessed])
    similarity_vector = cosine_similarity(index_TfIdf, index_query)
    similarity_vector = np.reshape(similarity_vector, -1)
    return similarity_vector


def computing_similarity_BM(index_BM, vectorizer_BM, query_preprocessed):
    index_query = vectorizer_BM.transform([query_preprocessed])
    scores = index_BM * index_query.T  # np.dot()
    similarity_vector = scores.toarray()  # вектор, i-й элемент которого обозначает близость запроса с
    # i-м документом корпуса
    return similarity_vector


def computing_similarity_BERT(index_BERT, query_embedding_array):  # подсчет близости запроса и документов корпуса
    similarity_vector = cosine_similarity(index_BERT, query_embedding_array)
    similarity_vector = np.reshape(similarity_vector, -1)
    return similarity_vector

