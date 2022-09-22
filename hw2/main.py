import argparse
import pickle
import pandas as pd
from pathlib import Path

import load_data
import preprocess_data
import compute_index
import compute_similarity


def main(query):
    vectorizer_path = Path(args.vectorizer_path)
    matrix_path = Path(args.matrix_path)
    texts, filepaths = load_data.data_loading(data_dir=args.data_dir)  # загрузка данных
    # Чтобы не индексировать корпус при каждом запуске программы, нужен следующий цикл:
    if vectorizer_path.is_file() and matrix_path.is_file():  # если векторайзер и матрица уже есть, то
        with open(vectorizer_path, 'rb') as file:  # они просто подгружаются
            vectorizer = pickle.load(file)
        with open(matrix_path, 'rb') as file:
            index_matrix = pickle.load(file)
    else:  # если векторайзер и матрица ещё не создавались, то
        texts_preprocessed = preprocess_data.data_preprocessing(texts)  # препроцессинг данных
        index_matrix, vectorizer = compute_index.computing_index_corpus(texts_preprocessed)  # индексация корпуса
        with open(vectorizer_path, 'wb') as file:  # векторайзер сохраняется
            pickle.dump(vectorizer, file)
        with open(matrix_path, 'wb') as file:  # матрица сохраняется
            pickle.dump(index_matrix, file)
    query_preprocessed = preprocess_data.query_preprocessing(query)  # препроцессинг запроса
    index_query = compute_index.computing_index_query(query_preprocessed, vectorizer)  # индексация запроса
    similarity_vector = compute_similarity.computing_similarity(index_matrix, index_query)  # подсчёт косинусной
    # близости между индексированным корпусом и запросом
    file_names = load_data.getting_filenames(filepaths)
    similarity_names = {'similarity': similarity_vector, 'doc_names': file_names}  # далее создаётся словарь,
    # который нужен для дальнейшего создания таблицы
    similarity_names_pd = pd.DataFrame(similarity_names)  # таблица нужна, чтобы выводить не только номера документов,
    # но и их названия
    similarity_names_pd = similarity_names_pd.sort_values(by=['similarity'], ascending=False)  # сортировка: вверху
    # аутпута находятся самые похожие на запрос документы
    print("Наиболее близкие к запросу документы: \n", similarity_names_pd['doc_names'])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_dir", help="Data directory")  # чтение пути как аргумента
    argparser.add_argument("matrix_path", help="Where to find or save an indexed corpus")  # чтение как аргумента
    # пути до файла-индексированного корпуса
    # NB! нужно дать на вход путь до папки + /index_matrix.pkl
    argparser.add_argument("vectorizer_path", help="Where to find or save a fitted vectorizer") # чтение как
    # аргумента пути до файла-векторайзера
    # NB! нужно дать на вход путь до папки + /vectorizer.pkl
    argparser.add_argument("query", help="Query")  # чтение запроса
    args = argparser.parse_args()
    main(query=args.query)
