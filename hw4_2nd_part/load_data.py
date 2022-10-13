import jsonlines


def get_filtered_answers(answers):
    def is_empty(x):  # есть ли что-то в строке
        return len(x['author_rating']['value']) != 0  # эта строчка возвращает значения True/False, проверяя,
    # есть ли что-то в value
    return list(filter(is_empty, answers))  # нас интересует только True, то есть когда в value что-то есть


def converted_answers_value(answers):
    for answer in answers:
        answer['author_rating']['value'] = int(answer['author_rating']['value'])  # значения ключа 'value' в данных
        # являются строковыми; чтобы в дальнейшем сортировать эти значения, их нужно преобразовать к типу int


def getting_answers(data_dir):
    result = []
    with jsonlines.open(data_dir) as file:
        for question in file:
            answers = question.get('answers')  # получение answers из данных
            if len(answers) == 0:
                continue

            answers = get_filtered_answers(answers)  # получение только тех ответов, значение value в которых не
            # является пустым
            converted_answers_value(answers)  # конвертация value в int
            answers.sort(key=lambda x: x['author_rating']['value'], reverse=True)  # сортировка ответ по value
            result.append(answers[0])  # добавление в список только тех ответов, которые имеют наибольший value

            if len(result) >= 10000:
                break
    answers = []
    for item in result:
        answers.append(item['text'])  # из ответов отбираются только тексты; упускается информация об авторе ответа

    return answers


def getting_questions(data_dir):
    result = []
    with jsonlines.open(data_dir) as file:
        for question in file:
            questions = question.get('question')  # получение question из данных
            if len(questions) == 0:
                continue

            result.append(questions)

            if len(result) >= 10000:
                break
    return result

