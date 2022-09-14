from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer


def computing_index_dictionary(texts_preprocessed):
    idxs_preprocessed = []
    for i, text_preprocessed in enumerate(texts_preprocessed):
        idxs_preprocessed.append(i)
    index_dict = {}
    tokenizer = RegexpTokenizer(r'\w+')
    for idx_preprocessed, text_preprocessed in zip(idxs_preprocessed, texts_preprocessed):
        text_preprocessed = tokenizer.tokenize(text_preprocessed)
        for word in text_preprocessed:
            if word in index_dict.keys():
                index_dict[word].append(idx_preprocessed)
            else:
                index_dict[word] = [idx_preprocessed]
    return index_dict


def computing_index_matrix(texts_preprocessed):
    vectorizer = CountVectorizer(analyzer='word')
    index_matrix = vectorizer.fit_transform(texts_preprocessed)
    return index_matrix, vectorizer

