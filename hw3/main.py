import argparse
import pickle
from pathlib import Path
import numpy as np

import load_data
import preprocess_data
import compute_index
import compute_similarity


def main(query):
    vectorizer_path = Path(args.vectorizer_path)
    matrix_path = Path(args.matrix_path)
    answers = load_data.getting_answers(data_dir=args.data_dir)  # загрузка ответов, которые пользователь получает
    # на выходе при запуске программы
    questions = load_data.getting_questions(data_dir=args.data_dir)  # загрузка вопросов: они нужны, чтобы осуществлять
    # поиск именно по ним
    # Чтобы не индексировать корпус при каждом запуске программы, нужен следующий цикл:
    if vectorizer_path.is_file() and matrix_path.is_file():  # если векторайзер и матрица уже есть, то
        with open(vectorizer_path, 'rb') as file:  # они просто подгружаются
            count_vectorizer = pickle.load(file)
        with open(matrix_path, 'rb') as file:
            matrix = pickle.load(file)
    else:  # если векторайзер и матрица ещё не создавались, то
        texts_preprocessed = preprocess_data.data_preprocessing(questions)  # препроцессинг данных
        matrix, count_vectorizer = compute_index.computing_index_corpus(texts_preprocessed)  # индексация корпуса
        with open(vectorizer_path, 'wb') as file:  # векторайзер сохраняется
            pickle.dump(count_vectorizer, file)
        with open(matrix_path, 'wb') as file:  # матрица сохраняется
            pickle.dump(matrix, file)
    query_preprocessed = preprocess_data.query_preprocessing(query)  # препроцессинг запроса
    query_count_vec = compute_index.computing_index_query(query_preprocessed, count_vectorizer)  # индексация запроса
    scores_vector = compute_similarity.computing_similarity(matrix, query_count_vec)  # подсчёт близости между
    # индексированным корпусом и запросом
    sorted_scores_indx = np.argsort(scores_vector, axis=0)[::-1]  # сортировка индексов скоров в обратном порядке
    # (по убыванию)
    corpus_doc_names = np.array(answers)
    sorted_corpus_doc_names = corpus_doc_names[sorted_scores_indx.ravel()]  # сортировка имён файлов
    # в соответствии со скорами
    print("Наиболее близкие к запросу ответы: \n", sorted_corpus_doc_names)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_dir", help="Data directory")  # чтение пути как аргумента
    argparser.add_argument("matrix_path", help="Where to find or save an indexed corpus")  # чтение как аргумента
    # пути до файла-индексированного корпуса
    # NB! нужно дать на вход путь до папки + /matrix.pkl
    argparser.add_argument("vectorizer_path", help="Where to find or save a fitted vectorizer")  # чтение как
    # аргумента пути до файла-count_vectorizer
    # NB! нужно дать на вход путь до папки + /count_vectorizer.pkl
    argparser.add_argument("query", help="Query")  # чтение запроса
    args = argparser.parse_args()
    main(query=args.query)

