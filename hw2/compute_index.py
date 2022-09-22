from sklearn.feature_extraction.text import TfidfVectorizer


def computing_index_corpus(texts_preprocessed):  # индексация корпуса предобработанных текстов с помощью
    # TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='word')
    index_matrix = vectorizer.fit_transform(texts_preprocessed)
    return index_matrix, vectorizer


def computing_index_query(query_preprocessed, vectorizer):  # индексация предобработанного запроса с помощью
    # TfidfVectorizer
    index_query = vectorizer.transform([query_preprocessed])
    return index_query
