import pandas as pd
import random
from sklearn.model_selection import train_test_split


class DataReader():

    def __init__(self):
        self._train_df = []
        self._val_df = []
        self._test_df = []

    def get_train_df(self):
        return self._train_df

    def get_test_df(self):
        return self._test_df

    def get_val_df(self):
        return self._val_df

    def read_data(self, file_handle, rand_seed):
        df = pd.read_csv(file_handle, sep=",")
        for col in df.columns[:-1]:
            df[col].fillna(df[col].median(), inplace=True)
            try:
                df[col] = pd.qcut(df[col], q=6, labels=False, duplicates='drop')
            except ValueError:
                df[col] = pd.cut(df[col], bins=6, labels=False)
            df[col] = df[col] / df[col].max()
        self._train_df, temp_df = train_test_split(df, test_size=0.30, random_state=rand_seed, shuffle=True)

        self._val_df, self._test_df = train_test_split(temp_df, test_size=0.50, random_state=rand_seed, shuffle=True)


def get_class_values(data_frame):
    return data_frame["quality"].tolist()


def get_labels_values(data_frame):
    return data_frame.drop(columns=["quality"]).tolist()


if __name__ == "__main__":
    reader = DataReader()
    r_seed = random.seed("1234")
    with open("./wsi5-25Z_dataset.csv") as file_h:
        reader.read_data(file_h, r_seed)
        print("Train:", reader.get_train_df().head())
        y_values = reader.get_train_df()["quality"].tolist()
        print("Y values:", y_values)
