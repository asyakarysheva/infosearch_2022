import numpy as np


def computing_index_answers(answers_embeddings):
    answers_embeddings_array = []
    for item in answers_embeddings:
        item = item.detach().numpy()
        answers_embeddings_array.append(item)
    matrix_answers = np.array(answers_embeddings_array)
    return matrix_answers


def computing_index_questions(questions_embeddings):
    questions_embeddings_array = []
    for item in questions_embeddings:
        item = item.detach().numpy()
        questions_embeddings_array.append(item)
    matrix_questions = np.array(questions_embeddings_array)
    return matrix_questions

