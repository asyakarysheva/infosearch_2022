"""
В этом модуле сортируются индексы скоров, а также имена файлов в соответствии со скорами.
"""

import numpy as np


def getting_results(similarity_vector, answers):
    sorted_scores_indx = np.argsort(similarity_vector, axis=0)[::-1]  # сортировка индексов скоров в обратном порядке
    # (по убыванию)
    corpus_doc_names = np.array(answers)
    sorted_corpus_doc_names = corpus_doc_names[sorted_scores_indx.ravel()]  # сортировка имён файлов
    # в соответствии со скорами
    return sorted_corpus_doc_names

