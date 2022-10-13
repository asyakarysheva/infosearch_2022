from sklearn.metrics.pairwise import cosine_similarity


def computing_similarity_BM(matrix, questions_count_matrix):
    scores = matrix*questions_count_matrix.T
    scores_matrix = scores.toarray()
    return scores_matrix


def computing_similarity_BERT(matrix_answers, matrix_questions):  # подсчета близости запроса и документов корпуса
    similarity_matrix = cosine_similarity(matrix_answers, matrix_questions)
    return similarity_matrix

