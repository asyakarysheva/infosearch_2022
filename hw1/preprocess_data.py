import ssl
import string

import nltk
from pymystem3 import Mystem

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

from nltk.corpus import stopwords


def load_mystem():
    mystem = Mystem()
    return mystem


def preprocess_text(texts, mystem):
    texts_preprocessed = []
    stop_rus = stopwords.words('russian')
    for text in texts:
        lemmatized = mystem.lemmatize(text)
        lemmatized = [w for w in lemmatized if w.strip() and w not in stop_rus]
        if lemmatized:
            texts = ' '.join(lemmatized)
            texts_preprocessed.append(texts)
    texts_wo_punct = []
    for text_preprocessed in texts_preprocessed:
        text_preprocessed.lower()
        for p in string.punctuation:
            if p in text_preprocessed:
                text_preprocessed = text_preprocessed.replace(p, '')
        texts_wo_punct.append(text_preprocessed)
    return texts_wo_punct


def data_preprocessing(texts):
    mystem = load_mystem()
    texts_preprocessed = preprocess_text(texts=texts, mystem=mystem)
    return texts_preprocessed
