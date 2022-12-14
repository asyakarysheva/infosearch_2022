"""
В этом модуле происходит предобработка запроса.
"""

import string
import nltk
from pymystem3 import Mystem

nltk.download('stopwords')

from nltk.corpus import stopwords


def load_mystem():  # загрузка Mystem
    mystem = Mystem()
    return mystem


def preprocess_query(query, mystem):
    query_lemmatized = ""  # лемматизация запроса (а также, получается, токенизация, поскольку mystem сразу
    # токенизирует)
    stop_rus = stopwords.words('russian')
    lemmatized = mystem.lemmatize(query)
    lemmatized = [w for w in lemmatized if w.strip() and w not in stop_rus]  # удаление стоп-слов
    if lemmatized:
        query_lemmatized = ' '.join(lemmatized)
    query_lemmatized.lower()
    query_wo_punct = query_lemmatized.translate(str.maketrans('', '', string.punctuation))  # вполне можно представить
    # запрос со знаками пунктуации (например, с запятой), поэтому можно удалить пунктуационные знаки
    return query_wo_punct


def query_preprocessing(query):  # функция, которая загружает Mystem и предобрабатывает запрос
    mystem = load_mystem()
    query_preprocessed = preprocess_query(query=query, mystem=mystem)
    return query_preprocessed

