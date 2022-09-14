import numpy as np


def task_dictionary(index_dict):
    max_len = max(len(value) for value in index_dict.values())
    for key, value in index_dict.items():
        if len(value) == max_len:
            word = key
            print('Самым частотным является слово:', word, '\n')

    least_frequent_words = []
    min_len = min(len(value) for value in index_dict.values())
    for key, value in index_dict.items():
        if len(value) == min_len:
            least_frequent_words.append(key)
    print('Самых редких слов', len(least_frequent_words), ', вот они, слева направо:', least_frequent_words, '\n')

    all_the_values = []
    for value in index_dict.values():
        for item in value:
            all_the_values.append(item)
    np.array(all_the_values)
    unique_values = np.unique(all_the_values)
    index_dict_unique = {}
    for key, value in index_dict.items():
        np.array(value)
        unique_value = np.unique(value)
        index_dict_unique[key] = unique_value
    words_in_all_docs = []
    for key, value in index_dict_unique.items():
        if np.array_equal(unique_values, value):
            words_in_all_docs.append(key)
    print('Слова, которые встречаются во всех документах:', words_in_all_docs, '\n')

    characters_count = {'Моника': len(index_dict['моника'] + index_dict['мон']),
                        'Рэйчел': len(index_dict['рэйчел'] + index_dict['рейч']),
                        'Чендлер': len(index_dict['чендлер'] + index_dict['чэндлер'] + index_dict['чен']),
                        'Фиби': len(index_dict['фиби'] + index_dict['фибс']),
                        'Росс': len(index_dict['росс']),
                        'Джоуи': len(index_dict['джоуи'] + index_dict['джо'])}
    # Я не искала в словаре "джои", потому что такого ключа в нём нет :(((
    print('Из главных героев статистически самый популярный (упоминается чаще всего):',
          max(characters_count, key=characters_count.get), '\n')


def task_matrix(index_matrix, vectorizer):
    matrix_freq = np.asarray(index_matrix.sum(axis=0)).ravel()
    final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])

    max_freq = np.max(matrix_freq)
    max_freq_ind = np.where(final_matrix[1] == str(max_freq))
    max_freq_word = final_matrix[0][max_freq_ind]
    print('Самым частотным является слово:', max_freq_word, '\n')

    min_freq = np.min(matrix_freq)
    min_freq_ind = np.where(final_matrix[1] == str(min_freq))
    min_freq_word = final_matrix[0][min_freq_ind]
    print('Самых редких слов', len(min_freq_word), ', вот они, слева направо:', min_freq_word, '\n')

    words_in_all_docs_ind = index_matrix.toarray().all(axis=0)
    print('Слова, которые встречаются во всех документах:', final_matrix[0][words_in_all_docs_ind], '\n')

    monica_ind = (final_matrix[0] == 'мон') | (final_matrix[0] == 'моника')
    monica_freq = final_matrix[1][monica_ind]
    monica_freq = list(map(int, monica_freq))
    monica_freq = sum(monica_freq)
    rach_ind = (final_matrix[0] == 'рэйчел') | (final_matrix[0] == 'рейч')
    rach_freq = final_matrix[1][rach_ind]
    rach_freq = list(map(int, rach_freq))
    rach_freq = sum(rach_freq)
    chan_ind = (final_matrix[0] == 'чендлер') | (final_matrix[0] == 'чэндлер') | (final_matrix[0] == 'чен')
    chan_freq = final_matrix[1][chan_ind]
    chan_freq = list(map(int, chan_freq))
    chan_freq = sum(chan_freq)
    phoebe_ind = (final_matrix[0] == 'фиби') | (final_matrix[0] == 'фибс')
    phoebe_freq = final_matrix[1][phoebe_ind]
    phoebe_freq = list(map(int, phoebe_freq))
    phoebe_freq = sum(phoebe_freq)
    ross_ind = final_matrix[0] == 'росс'
    ross_freq = final_matrix[1][ross_ind]
    ross_freq = list(map(int, ross_freq))
    ross_freq = sum(ross_freq)
    joey_ind = (final_matrix[0] == 'джоуи') | (final_matrix[0] == 'джо')
    joey_freq = final_matrix[1][joey_ind]
    joey_freq = list(map(int, joey_freq))
    joey_freq = sum(joey_freq)
    characters_count = {'Моника': monica_freq,
                        'Рэйчел': rach_freq,
                        'Чендлер': chan_freq,
                        'Фиби': phoebe_freq,
                        'Росс': ross_freq,
                        'Джоуи': joey_freq}
    print('Из главных героев статистически самый популярный (упоминается чаще всего):',
          max(characters_count, key=characters_count.get), '\n')
