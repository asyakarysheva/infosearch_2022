import string

import nltk
from pymystem3 import Mystem

nltk.download('stopwords')

from nltk.corpus import stopwords


def load_mystem():  # загрузка Mystem
    mystem = Mystem()
    return mystem


def preprocess_text(texts, mystem):
    texts_preprocessed = []
    stop_rus = stopwords.words('russian')
    for text in texts:  # лемматизация текстов (а также, получается, токенизация, поскольку mystem сразу токенизирует)
        lemmatized = mystem.lemmatize(text)
        lemmatized = [w for w in lemmatized if w.strip() and w not in stop_rus]  # удаление стоп-слов
        if lemmatized:
            texts = ' '.join(lemmatized)
            texts_preprocessed.append(texts)
    texts_wo_punct = []
    for text_preprocessed in texts_preprocessed:
        text_preprocessed.lower()  # приведение к нижнему регистру
        for p in string.punctuation:  # удаление пунктуации
            if p in text_preprocessed:
                text_preprocessed = text_preprocessed.replace(p, '')
        texts_wo_punct.append(text_preprocessed)
    return texts_wo_punct


def data_preprocessing(texts):  # функция, которая которая загружает Mystem и предобрабатывает тексты
    mystem = load_mystem()
    texts_preprocessed = preprocess_text(texts=texts, mystem=mystem)
    return texts_preprocessed


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
