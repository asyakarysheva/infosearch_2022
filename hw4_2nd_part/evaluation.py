def evaluating_quality(sorted_scores_indx):
    columns = sorted_scores_indx.shape[1]  # количество колонок в матрице
    if_answer_in_top_5_list = []  # список из элементов, каждый из которых содержит в себе пять значений True/False
    for i in range(columns):  # для каждого запроса проверяется, есть ли его ответ в топ-5 результатов
        top_5 = sorted_scores_indx[:, i][0:5]
        if_answer_in_top_5 = top_5 == i  # вопрос и ответ на него имеют одинаковые индексы, поэтому проверяется, есть
        # ли в топ-5 результатов элемент с таким же индексом, как у вопроса
        if_answer_in_top_5_list.append(if_answer_in_top_5)
    number_of_successful_results = 0  # в эту переменную записывается количество вопросов, в топ-5 результатов для
    # которых есть их ответ
    for item in if_answer_in_top_5_list:  # итерация по списку, созданному чуть ранее
        if True in item:
            number_of_successful_results += 1
    metric_value = number_of_successful_results / columns  # итоговая метрика качества
    # она равна сумме скоров для каждого запроса (number_of_successful_results), деленной на количество запросов
    return metric_value

