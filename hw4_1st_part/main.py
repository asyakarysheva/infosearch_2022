# В этом модуле закомменчены те части кода, которые нужны только для того, чтобы подготовить данные для
# векторизации. Например, препроцессинг ответов и сохранение предобработанных ответов в отдельный файл.

# Я не предполагаю, что нужно будет раскомменчивать закомменченное, при этом не имея файла с готовыми эмбеддингами:
# по сути закоменченное нужно только для воспроизводимости.

import argparse
import numpy as np
# Следующую библиотеку нужно раскомментить, если нужно предобработать ответы
# from pathlib import Path

import load_data
import preprocess_data
import compute_index
import compute_similarity


def main(query):
    answers = load_data.getting_answers(data_dir=args.data_dir)  # загрузка ответов, которые пользователь получает
    # на выходе при запуске программы

    # Строчка ниже закомменчена, потому что предобработанные ответы понадобятся только в том случае, если ещё не
    # посчитаны их эмбеддинги
    # texts_preprocessed = preprocess_data.data_preprocessing(answers)

    # Следующие несколько строк тоже закомменчены: они нужны для сохранения предобработанных ответов
    # texts_preprocessed_path = Path(args.texts_preprocessed_path)
    # with open(texts_preprocessed_path, 'w') as fp:
       # for text_preprocessed in texts_preprocessed:
           # fp.write("%s\n" % text_preprocessed)

    query_preprocessed = preprocess_data.query_preprocessing(query)  # предобработка запроса

    matrix = compute_index.computing_index_corpus(embeddings_path=args.embeddings_path)  # индексированный корпус
    query_embedding_array = compute_index.computing_index_query(query_preprocessed)
    similarity_vector = compute_similarity.computing_similarity(matrix, query_embedding_array)
    sorted_scores_indx = np.argsort(similarity_vector, axis=0)[::-1]  # сортировка индексов скоров в обратном порядке
    # (по убыванию)
    corpus_doc_names = np.array(answers)
    sorted_corpus_doc_names = corpus_doc_names[sorted_scores_indx.ravel()]  # сортировка имён файлов
    # в соответствии со скорами
    print("Наиболее близкие к запросу ответы: \n", sorted_corpus_doc_names[:10])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_dir", help="Data directory")  # чтение пути к данным
    # Строчка ниже закомменчена, её нужно раскомментить, только если эмбеддинги ещё не были получены в Colab
    # argparser.add_argument("texts_preprocessed_path", help="Where to save preprocessed answers")
    # NB! нужно дать на вход путь до папки + /texts_preprocessed.txt
    argparser.add_argument("embeddings_path", help="Where to find or save an indexed corpus")  # чтение как аргумента
    # пути до файла-индексированного корпуса (до эмбеддингов ответов корпуса)
    argparser.add_argument("query", help="Query")  # чтение запроса
    args = argparser.parse_args()
    main(query=args.query)
