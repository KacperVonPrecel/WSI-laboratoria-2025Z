from id3_algorithm import DecisionSolver
from data_reader import DataReader
from matplotlib import pyplot as plt

MIN_DEPTH = 1
MAX_DEPTH = 12

if __name__ == "__main__":
    print("Input csv file with data:")
    data_set = str(input())
    with open(data_set) as file_h:
        reader = DataReader()
        solver = DecisionSolver()
        reader.read_data(file_h)
        target = reader.get_train_df().columns.tolist()[-1]

    generated_trees = []
    accuracy_list = []
    print("Fitting ...")
    for depth in range(MIN_DEPTH, MAX_DEPTH):
        generated_trees.append(solver.fit(reader.get_train_df(), depth))

    for model in generated_trees:
        accuracy_list.append(round((solver.predict(reader.get_val_df(), model, target) * 100), 2))

    best_accuracy = max(accuracy_list)
    best_model = generated_trees[accuracy_list.index(best_accuracy)]

    final_accuracy = round((solver.predict(reader.get_test_df(), best_model, target) * 100), 2)

    print(f"Best depth: {generated_trees.index(best_model) + 1}")
    print(f"Best accuracy: {final_accuracy}")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    x_labels = [i for i in range(MIN_DEPTH, MAX_DEPTH)]
    ax.set_xlabel("Tree depth", fontsize=18)
    ax.set_ylabel("Accuracy in %", rotation=90, fontsize=18)
    ax.set_xticks(x_labels)

    tp = ax.bar(x_labels, accuracy_list, color="blue")

    plt.title("Dokładność modelu dla wybranych głębokości", fontweight='bold', fontsize=26)
    plt.savefig("Dokładność modelu dla wybranych głębokości")
    plt.show()

    print("Done")
