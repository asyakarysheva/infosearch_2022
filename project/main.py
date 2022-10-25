import streamlit as st
import os
import time

from load_data import answers
import query_preprocess
import compute_index_query_BERT
import compute_similarity
import get_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # чтобы не показывались предупреждения о tokenizer для BERT


def TfIdf_search(query_preprocessed):
    from load_indexed_corpus_and_models import index_TfIdf, vectorizer_TfIdf
    similarity_vector = compute_similarity.computing_similarity_TfIdf(index_TfIdf, vectorizer_TfIdf, query_preprocessed)
    sorted_corpus_doc_names = get_results.getting_results(similarity_vector, answers)
    return sorted_corpus_doc_names


def BM25_search(query_preprocessed):
    from load_indexed_corpus_and_models import index_BM, vectorizer_BM
    similarity_vector = compute_similarity.computing_similarity_BM(index_BM, vectorizer_BM, query_preprocessed)
    sorted_corpus_doc_names = get_results.getting_results(similarity_vector, answers)
    return sorted_corpus_doc_names


def BERT_search(query_preprocessed):
    from load_indexed_corpus_and_models import index_BERT, tokenizer_BERT, model_BERT
    query_embedding_array = compute_index_query_BERT.computing_index_query(query_preprocessed, tokenizer_BERT, model_BERT)
    similarity_vector = compute_similarity.computing_similarity_BERT(index_BERT, query_embedding_array)
    sorted_corpus_doc_names = get_results.getting_results(similarity_vector, answers)
    return sorted_corpus_doc_names


if __name__ == '__main__':

    st.title('Проект по инфопоиску')
    st.header('Поисковик с тремя методами поиска')
    st.info('Здесь можно найти наиболее похожие на запрос ответы на вопросы с mail.ru, посвящённые любви. Для того, чтобы '
        'начать поиск, нужно ввести запрос, выбрать метод поиска и нажать на кнопку для запуска поиска по введённому запросу.')
    st.image("https://upload.wikimedia.org/wikipedia/ru/thumb/8/81/ОтветыMail.ru.svg/1200px-ОтветыMail.ru.svg.png")

    query = st.text_input('Пожалуйста, введите запрос:')
    search_algo = st.selectbox('Пожалуйста, выберите метод поиска:', ['TF-IDF', 'BM25', 'BERT'])
    start = time.time()
    query_preprocessed = query_preprocess.query_preprocessing(query)

    if st.button('Начать поиск'):
        if search_algo == 'TF-IDF':
            sorted_corpus_doc_names = TfIdf_search(query_preprocessed)
        elif search_algo == 'BM25':
            sorted_corpus_doc_names = BM25_search(query_preprocessed)
        else:
            sorted_corpus_doc_names = BERT_search(query_preprocessed)
        st.write("Наиболее близкие к запросу ответы:")
        st.table(sorted_corpus_doc_names[:10])
        end = time.time()
        st.write("На поиск потрачено столько секунд: \n", end - start)
