import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def computing_similarity(index_matrix, index_query):  # подсчета близости запроса и документов корпуса
    similarity_vector = cosine_similarity(index_matrix, index_query)
    similarity_vector = np.reshape(similarity_vector, -1)
    return similarity_vector
