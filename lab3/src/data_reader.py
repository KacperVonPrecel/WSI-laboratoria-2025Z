import pandas as pd
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

    def read_data(self, file_handle):
        df = pd.read_csv(file_handle, sep=";")

        for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
            df[col].fillna(df[col].mode()[0], inplace=True)

        for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
            df[col].fillna(df[col].median(), inplace=True)

        df["age"] = pd.qcut(df["age"], q=4, labels=[1, 2, 3, 4])

        df = df[(df["height"] > 0) & (df["height"] < 300)]
        df["height"] = pd.qcut(df["height"], q=4, labels=[1, 2, 3, 4])

        df = df[(df["weight"] > 0) & (df["weight"] < 300)]
        df["weight"] = pd.qcut(df["weight"], q=4, labels=[1, 2, 3, 4])

        df = df[(df["ap_hi"] > 0) & (df["ap_hi"] < 300)]
        df["ap_hi"] = pd.cut(df["ap_hi"], bins=4, labels=[1, 2, 3, 4])

        df = df[(df["ap_lo"] > 0) & (df["ap_lo"] < 300)]
        df["ap_lo"] = pd.cut(df["ap_lo"], bins=4, labels=[1, 2, 3, 4])

        self._train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)

        self._val_df, self._test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)


def get_class_values(data_frame):
    return data_frame["cardio"].tolist()


def get_labels_values(data_frame):
    return data_frame.drop(columns=["cardio"]).tolist()


# if __name__ == "__main__":
#     reader = DataReader()
#     with open("./cardio_train.csv") as file_h:
#         reader.read_data(file_h)
#         print("Train:", reader.get_train_df().head())
#         y_values = reader.get_train_df()["cardio"].tolist()
#         print("Y values:", y_values)
