import string

import nltk
from pymystem3 import Mystem

nltk.download('stopwords')

from nltk.corpus import stopwords


def load_mystem():  # загрузка Mystem
    mystem = Mystem()
    return mystem


def preprocess_answers(answers, mystem):
    texts_preprocessed = []
    stop_rus = stopwords.words('russian')
    for answer in answers:  # лемматизация текстов (а также, получается, токенизация, поскольку mystem сразу токенизирует)
        lemmatized = mystem.lemmatize(answer)
        lemmatized = [w for w in lemmatized if w.strip() and w not in stop_rus]  # удаление стоп-слов
        if lemmatized:
            answers = ' '.join(lemmatized)
            texts_preprocessed.append(answers)
    texts_wo_punct = []
    for text_preprocessed in texts_preprocessed:
        text_preprocessed.lower()  # приведение к нижнему регистру
        for p in string.punctuation:  # удаление пунктуации
            if p in text_preprocessed:
                text_preprocessed = text_preprocessed.replace(p, '')
        texts_wo_punct.append(text_preprocessed)
    return texts_wo_punct


def answers_preprocessing(answers):  # функция, которая которая загружает Mystem и предобрабатывает тексты
    mystem = load_mystem()
    answers_preprocessed = preprocess_answers(answers=answers, mystem=mystem)
    return answers_preprocessed


def preprocess_questions(questions, mystem):
    texts_preprocessed = []
    stop_rus = stopwords.words('russian')
    for question in questions:  # лемматизация текстов (а также, получается, токенизация, поскольку mystem сразу токенизирует)
        lemmatized = mystem.lemmatize(question)
        lemmatized = [w for w in lemmatized if w.strip() and w not in stop_rus]  # удаление стоп-слов
        if lemmatized:
            questions = ' '.join(lemmatized)
            texts_preprocessed.append(questions)
    texts_wo_punct = []
    for text_preprocessed in texts_preprocessed:
        text_preprocessed.lower()  # приведение к нижнему регистру
        for p in string.punctuation:  # удаление пунктуации
            if p in text_preprocessed:
                text_preprocessed = text_preprocessed.replace(p, '')
        texts_wo_punct.append(text_preprocessed)
    return texts_wo_punct


def questions_preprocessing(questions):  # функция, которая загружает Mystem и предобрабатывает запрос
    mystem = load_mystem()
    questions_preprocessed = preprocess_questions(questions=questions, mystem=mystem)
    return questions_preprocessed

