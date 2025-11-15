from id3_algorithm import DecisionSolver
from data_reader import DataReader

if __name__ == "__main__":
    print("Input csv file with data:")
    file = str(input())
    with open(file) as file_h:
        reader = DataReader()
        solver = DecisionSolver()
        reader.read_data(file_h)

        test = solver.fit(reader.get_train_df(), 8)
        print(test.children.keys())
        print("Done")