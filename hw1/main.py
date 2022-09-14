import argparse

import compute_index
import load_data
import preprocess_data
import task_hw1

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_dir", help="Data directory")  # чтение пути как аргумента
    argparser.add_argument("index_type", help="Index format (dictionary or matrix)")  # чтение формата индекса как
    # аргумента
    args = argparser.parse_args()
    texts = load_data.data_loading(data_dir=args.data_dir)  # загрузка данных
    texts_preprocessed = preprocess_data.data_preprocessing(texts)  # препроцессинг данных
    if args.index_type != "dictionary" and args.index_type != "matrix":  # что делать, если второй аргумент – ни
        # dictionary, ни matrix
        print("Выберите либо 'dictionary', либо 'matrix' как формат обратного индекса")
    if args.index_type == "dictionary":
        index_dict = compute_index.computing_index_dictionary(texts_preprocessed)  # создание индекса в формате словаря
        task_hw1.task_dictionary(index_dict)  # выполнение задания
    if args.index_type == "matrix":
        index_matrix, vectorizer = compute_index.computing_index_matrix(texts_preprocessed)  # создание индекса в
        # виде матрицы
        task_hw1.task_matrix(index_matrix, vectorizer)  # выполнение задания
