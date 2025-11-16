from id3_algorithm import DecisionSolver
from data_reader import DataReader

if __name__ == "__main__":
    print("Input csv file with data:")
    data_set = str(input())
    with open(data_set) as file_h:
        reader = DataReader()
        solver = DecisionSolver()
        reader.read_data(file_h)
        target = reader.get_train_df().columns.tolist()[-1]

        print("Fitting ...")
        test = solver.fit(reader.get_train_df(), 3)
        accuracy = solver.predict(reader.get_test_df(), test, target)
        print(f"Accuracy: {accuracy}")
        print(test.children.keys())
        print("Done")