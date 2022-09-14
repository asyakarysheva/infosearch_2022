from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer


def computing_index_dictionary(texts_preprocessed):
    idxs_preprocessed = []
    for i, text_preprocessed in enumerate(texts_preprocessed):  # создание списка с индексами (номерами) текстов
        idxs_preprocessed.append(i)
    index_dict = {}  # создание словаря-будущего индекса
    tokenizer = RegexpTokenizer(r'\w+')
    for idx_preprocessed, text_preprocessed in zip(idxs_preprocessed, texts_preprocessed):
        text_preprocessed = tokenizer.tokenize(text_preprocessed)  # деление текстов на слова
        for word in text_preprocessed:
            if word in index_dict.keys():  # если слово уже есть в ключах, то
                index_dict[word].append(idx_preprocessed)  # к существующим значениям добавляются ещё значения
            else:  # если слова нет в ключах,
                index_dict[word] = [idx_preprocessed]  # то текущий номер слова в тексте присваивается значению ключа
    return index_dict


def computing_index_matrix(texts_preprocessed):  # создание матрицы как индекса с помощью CountVectorizer
    vectorizer = CountVectorizer(analyzer='word')
    index_matrix = vectorizer.fit_transform(texts_preprocessed)
    return index_matrix, vectorizer  # нужно вернуть vectorizer, чтобы затем иметь доступ именно к словам

