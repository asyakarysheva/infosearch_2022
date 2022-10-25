"""
Этот модуль нужен для того, чтобы векторизовать запрос с помощью BERT.
"""

import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def computing_index_query(query_preprocessed, tokenizer_BERT, model_BERT):  # индексация запроса
    # Токенизация запроса
    encoded_input = tokenizer_BERT(query_preprocessed, padding=True, truncation=True, max_length=24, return_tensors='pt')
    # Получение эмбеддингов для запроса
    with torch.no_grad():
        model_output = model_BERT(**encoded_input)
    # Выполнение mean pooling
    query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    # Преобразование к классу numpy.ndarray
    query_embedding_array = query_embedding.numpy()
    return query_embedding_array

