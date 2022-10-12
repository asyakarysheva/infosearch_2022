import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def computing_similarity(matrix, query_embedding_array):  # подсчет близости запроса и документов корпуса
    similarity_vector = cosine_similarity(matrix, query_embedding_array)
    similarity_vector = np.reshape(similarity_vector, -1)
    return similarity_vector

