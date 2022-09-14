import ssl
import string

import nltk
from pymystem3 import Mystem

# Всё, что следует ниже (до nltk.download(...) невключительно), было добавлено мной, чтобы избежать ошибки,
# связанной с nltk (насколько помню, именно с загрузкой stopwords), на моём компьютере
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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


def data_preprocessing(texts):  # функция, которая и Mystem загружает, и тексты предобрабатывает
    mystem = load_mystem()
    texts_preprocessed = preprocess_text(texts=texts, mystem=mystem)
    return texts_preprocessed
