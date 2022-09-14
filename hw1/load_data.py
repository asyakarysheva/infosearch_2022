import os


def data_loading(data_dir):
    filepaths = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name != '.DS_Store':
                filepaths.append(os.path.join(root, name))
    texts = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8-sig') as f:  # encoding='utf-8-sig', чтобы не возникало \ufeff в аутпутах
            texts.append(f.read())
    for text in texts:
        text.strip()
    print(len(texts), 'текстов в выборке')
    return texts

