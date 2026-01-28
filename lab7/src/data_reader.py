import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np


class DataReader():

    def __init__(self):
        self._train_X = None
        self._train_y = None
        self._val_X = None
        self._val_y = None
        self._test_X = None
        self._test_y = None

    def get_train_x(self):
        return self._train_X

    def get_train_y(self):
        return self._train_y

    def get_test_x(self):
        return self._test_X

    def get_test_y(self):
        return self._test_y

    def get_val_x(self):
        return self._val_X

    def get_val_y(self):
        return self._val_y

    def read_data(self, file_handle, rand_seed):
        df = pd.read_csv(file_handle, sep=",", encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']

        self.vocab = create_vocabulary(df['message'])
        text_data = text_to_vector(df['message'], self.vocab)
        label_data = df['label'].values

        self._train_X, temp_text_data, self._train_y, temp_label_data = train_test_split(
            text_data,
            label_data,
            test_size=0.30,
            random_state=rand_seed,
            shuffle=True)

        self._val_X, self._test_X, self._val_y, self._test_y = train_test_split(
            temp_text_data,
            temp_label_data,
            test_size=0.50,
            random_state=rand_seed,
            shuffle=True)


def preprocess_text(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def create_vocabulary(texts: list[str]):
    vocabulary = set()
    for text in texts:
        tokens = preprocess_text(text)
        vocabulary.update(tokens)
    return sorted(list(vocabulary))


def text_to_vector(texts: list[str], vocabulary: list[str]):
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=int)
    for i, text in enumerate(texts):
        tokens = preprocess_text(text)
        for token in tokens:
            if token in word_to_idx:
                matrix[i, word_to_idx[token]] += 1
    return matrix
