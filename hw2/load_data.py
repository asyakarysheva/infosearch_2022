import os
import re


def data_loading(data_dir):
    filepaths = []
    for root, dirs, files in os.walk(data_dir):  # чтение путей до файла
        for name in files:
            if name != '.DS_Store':
                filepaths.append(os.path.join(root, name))
    texts = []
    for filepath in filepaths:  # чтение самих текстов
        with open(filepath, 'r',
                  encoding='utf-8-sig') as f:  # encoding='utf-8-sig', чтобы не возникало \ufeff в аутпутах
            texts.append(f.read())
    return texts, filepaths


def getting_filenames(filepaths):
    file_names = []
    for file in filepaths:
        file_names.append(re.findall('(?<=/)[^/]+(?=\.txt$)', file)[0])  # нужно найти всё, что предшествует .txt в пути
        # до файла – это и будет названием документа
    return file_names
