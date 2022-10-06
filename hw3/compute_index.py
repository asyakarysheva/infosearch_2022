from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

# необходимые константы
k = 2
b = 0.75


def computing_index_corpus(texts_preprocessed):  # индексация корпуса
    # создание матрицы tf + понадобится для индексации запроса
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(texts_preprocessed)
    tf = count
    # для расчета idf
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(texts_preprocessed)
    idf = tfidf_vectorizer.idf_
    # подсчёт количества слов в документах
    len_d = tf.sum(axis=1)
    # подсчёт среднего количества слов документа в корпусе
    avdl = len_d.mean()
    values = []  # лист из n значений, которые мы хотим положить в матрицу
    rows = []  # лист из n значений, где i-тое значение это индекс строки i-го элемента из values
    cols = []  # лист из n значений, где i-тое значение это индекс колонки i-го элемента из values
    for i, j in zip(*tf.nonzero()):  # итерация по ненулевым значениям матрицы tf
        cur_tf = tf[i, j]
        cur_idf = idf[j]
        cur_len_d = len_d[i]
        # расчёт числителя
        A = cur_idf * cur_tf * (k + 1)
        # расчёт знаменателя
        B_1 = (k * (1 - b + b * cur_len_d / avdl))
        B = cur_tf + B_1
        values.append((A / B).item())  # получение значений BM25 для итоговой матрицы
        rows.append(i)  # получение индекса строки
        cols.append(j)  # получение индекса колонки
    matrix = sparse.csr_matrix((values, (rows, cols)))  # создание матрицы
    return matrix, count_vectorizer


def computing_index_query(query_preprocessed, count_vectorizer):  # индексация запроса
    query_count_vec = count_vectorizer.transform([query_preprocessed])
    return query_count_vec

