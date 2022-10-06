def computing_similarity(matrix, query_count_vec):  # подсчёт близости запроса и документов корпуса
    scores = matrix * query_count_vec.T
    scores_vector = scores.toarray()  # вектор, i-й элемент которого обозначает близость запроса с
    # i-м документом корпуса
    return scores_vector

