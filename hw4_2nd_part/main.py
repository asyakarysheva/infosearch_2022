import argparse
import pickle
from pathlib import Path
import numpy as np
import torch

import load_data
import preprocess_data
import compute_index_BM
import compute_index_BERT
import compute_similarity
import evaluation


def BM_evaluation():
    answers = load_data.getting_answers(data_dir=args.data_dir)  # загрузка ответов
    questions = load_data.getting_questions(data_dir=args.data_dir)  # загрузка вопросов
    answers_matrix_path = Path(args.answers_matrix_path)
    questions_matrix_path = Path(args.questions_matrix_path)
    if answers_matrix_path.is_file() and questions_matrix_path.is_file():  # если матрица с ответами и матрица
        # с вопросами уже есть, то
        with open(answers_matrix_path, 'rb') as file:  # они просто подгружаются
            answers_matrix = pickle.load(file)
        with open(questions_matrix_path, 'rb') as file:
            questions_matrix = pickle.load(file)
    else:  # если векторайзер и матрица ещё не создавались, то
        answers_preprocessed = preprocess_data.answers_preprocessing(answers)  # препроцессинг ответов
        answers_matrix, count_vectorizer = compute_index_BM.computing_index_answers(answers_preprocessed)  # индексация
        # ответов
        questions_preprocessed = preprocess_data.questions_preprocessing(questions)  # препроцессинг вопросов
        questions_matrix = compute_index_BM.computing_index_questions(questions_preprocessed, count_vectorizer)
        with open(answers_matrix_path, 'wb') as file:  # векторизатор сохраняется
            pickle.dump(answers_matrix, file)
        with open(questions_matrix_path, 'wb') as file:  # матрица сохраняется
            pickle.dump(questions_matrix, file)
    scores_matrix = compute_similarity.computing_similarity_BM(answers_matrix, questions_matrix)  # подсчёт близости
    # между ответами и вопросами
    sorted_scores_indx = np.argsort(scores_matrix, axis=0)[::-1]  # сортировка индексов скоров в обратном порядке
    metric_value = evaluation.evaluating_quality(sorted_scores_indx)  # подсчёт метрики качества
    return metric_value


def BERT_evaluation():
    # Ниже закомменчено много строк: они нужны для того, чтобы предобработать данные,
    # а затем дать эти данные на вход модели в Colab
    # answers = load_data.getting_answers(data_dir=args.data_dir)  # загрузка ответов
    # questions = load_data.getting_questions(data_dir=args.data_dir)  # загрузка вопросов

    # Сохранение предобработанных ответов
    # answers_preprocessed = preprocess_data.answers_preprocessing(answers)
    # answers_preprocessed_path = Path(args.answers_preprocessed_path)
    # with open(answers_preprocessed_path, 'w') as fp:
        # for answer_preprocessed in answers_preprocessed:
            # fp.write("%s\n" % answer_preprocessed)

    # Сохранение предобработанных вопросов
    # questions_preprocessed = preprocess_data.questions_preprocessing(questions)
    # questions_preprocessed_path = Path(args.questions_preprocessed_path)
    # with open(questions_preprocessed_path, 'w') as fp:
        # for question_preprocessed in questions_preprocessed:
        # fp.write("%s\n" % question_preprocessed)

    # Загрузка посчитанных в Colab эмбеддингов
    answers_embeddings = torch.load(args.answers_embeddings_path, map_location=torch.device('cpu'))
    questions_embeddings = torch.load(args.questions_embeddings_path, map_location=torch.device('cpu'))
    # Приведение полученных эмбеддингов к формату денс матриц для дальнейшего вычисления близости
    matrix_answers = compute_index_BERT.computing_index_answers(answers_embeddings)
    matrix_questions = compute_index_BERT.computing_index_questions(questions_embeddings)
    # Подсчёт косинусной близости между матрицами
    similarity_matrix = compute_similarity.computing_similarity_BERT(matrix_answers, matrix_questions)
    sorted_scores_indx = np.argsort(similarity_matrix, axis=0)[::-1]  # сортировка индексов скоров в обратном порядке
    metric_value_BERT = evaluation.evaluating_quality(sorted_scores_indx)
    return metric_value_BERT  # подсчёт метрики качества


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_dir", help="Data directory")  # чтение пути как аргумента
    argparser.add_argument("answers_matrix_path", help="Where to find or save indexed answers")  # чтение пути до
    # файла с индексированными ответами BM25
    # NB! нужно дать на вход путь до папки + /answers_matrix.pkl
    argparser.add_argument("questions_matrix_path", help="Where to find or save indexed questions")  # чтение пути до
    # файла с индексированными вопросами
    # NB! нужно дать на вход путь до папки + /questions_matrix.pkl
    argparser.add_argument("answers_embeddings_path", help="Where to find embeddings for answers")  # чтение
    # пути до эмбеддингов ответов
    argparser.add_argument("questions_embeddings_path", help="Where to find embeddings for questions")  # чтение пути
    # до эмбеддингов вопросов
    # Строчка ниже закомменчены, их нужно раскомментить, только если эмбеддинги ещё не были получены в Colab
    # argparser.add_argument("answers_preprocessed_path", help="Where to save preprocessed answers")
    # NB! нужно дать на вход путь до папки + /answers_preprocessed.txt
    # argparser.add_argument("questions_preprocessed_path", help="Where to save preprocessed questions")
    # NB! нужно дать на вход путь до папки + /questions_preprocessed.txt
    args = argparser.parse_args()
    metric_value_BM = BM_evaluation()
    metric_value_BERT = BERT_evaluation()
    print("Значение метрики качества для BM:", metric_value_BM, "\n", "Значение метрики качества для BERT:", metric_value_BERT)
