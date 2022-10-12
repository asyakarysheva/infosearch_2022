# В этом модуле не происходит индексации корпуса: индексация выполняется в Google Colab с GPU.
# Код для индексации корпуса можно найти в файле hw4_vectorization.ipynb.
# Здесь же только загружается список из эмбеддингов, полученный в Colab.

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def computing_index_corpus(embeddings_path):
    answers_embeddings = torch.load(embeddings_path, map_location=torch.device('cpu'))
    # Загруженный из Colab файл представляет собой список из 50 тысяч элементов типа torch.Tensor
    # Чтобы сделать из списка матрицу, выполняется следующий код:
    # (я не уверена, что это наиболее эффективный способ построения матрицы формата Document-Term, но он как минимум
    # рабочий)
    answers_embeddings_array = []
    for item in answers_embeddings:
        item = item.detach().numpy()  # элементы списка приводятся к классу numpy.ndarray
        answers_embeddings_array.append(item)
    # Далее создаётся матрица типа numpy.ndarray:
    # Её размер – (50000, 1024)
    matrix = np.array(answers_embeddings_array)
    return matrix


# Запрос же векторизуется здесь, а не в Colab.


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return tokenizer


def load_model():
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return model


def computing_index_query(query_preprocessed):  # индексация запроса
    tokenizer = load_tokenizer()
    model = load_model()
    # Токенизация запроса
    encoded_input = tokenizer(query_preprocessed, padding=True, truncation=True, max_length=24, return_tensors='pt')
    # Получение эмбеддингов для запроса
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Выполнение mean pooling
    query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    # Преобразование к классу numpy.ndarray
    query_embedding_array = query_embedding.detach().numpy()
    return query_embedding_array

