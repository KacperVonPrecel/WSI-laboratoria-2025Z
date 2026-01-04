import pandas as pd
import random
from sklearn.model_selection import train_test_split


class DataReader():

    def __init__(self):
        self._train_X = []
        self._train_y = []
        self._val_X = []
        self._val_y = []
        self._test_X = []
        self._test_y = []

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
        df = pd.read_csv(file_handle, sep=",")
        for col in df.columns[:-1]:
            df[col].fillna(df[col].median(), inplace=True)
            try:
                df[col] = pd.qcut(df[col], q=6, labels=False, duplicates='drop')
            except ValueError:
                df[col] = pd.cut(df[col], bins=6, labels=False)
            df[col] = df[col] / df[col].max()

        labels_data = get_labels_values(df)
        y_data = get_class_values(df)
        temp_x, self._train_X, temp_y, self._train_y = train_test_split(
            labels_data,
            y_data,
            test_size=0.70,
            random_state=rand_seed,
            shuffle=True)

        self._val_X, self._test_X, self._val_y, self._test_y = train_test_split(
            temp_x,
            temp_y,
            test_size=0.50,
            random_state=rand_seed,
            shuffle=True)


def get_class_values(data_frame: pd.DataFrame):
    return pd.get_dummies(data_frame.iloc[:, -1]).to_numpy().astype(float)


def get_labels_values(data_frame: pd.DataFrame):
    return data_frame.iloc[:, :-1].to_numpy().astype(float)


# if __name__ == "__main__":
#     reader = DataReader()
#     r_seed = random.seed("1234")
#     with open("./wsi5-25Z_dataset.csv") as file_h:
#         reader.read_data(file_h, r_seed)
#         print("Train:", reader.get_train_df().head())
#         y_values = reader.get_train_df()["quality"].tolist()
#         print("Y values:", y_values)
