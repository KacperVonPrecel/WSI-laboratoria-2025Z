from id3_algorithm import DecisionSolver
from random_forest import RandomForest
from data_reader import DataReader
from matplotlib import pyplot as plt

MIN_DEPTH = 1
MAX_DEPTH = 12

TREE_NUMS = [200, 400, 600, 800, 1000, 1200, 1400, 1600]

SETS_RAND_SEED = [38, 42, 46, 50, 54]

def testing_id3(reader: DataReader, solver: DecisionSolver, target: str):
    generated_trees = []
    accuracy_list = []
    print("Fitting ...")
    for depth in range(MIN_DEPTH, MAX_DEPTH):
        generated_trees.append(solver.fit(reader.get_train_df(), depth))

    for model in generated_trees:
        accuracy_list.append(round((solver.check_accuracy(reader.get_val_df(), model, target) * 100), 2))

    best_accuracy = max(accuracy_list)
    best_model = generated_trees[accuracy_list.index(best_accuracy)]

    final_accuracy = round((solver.check_accuracy(reader.get_test_df(), best_model, target) * 100), 2)

    print(f"Best depth: {generated_trees.index(best_model) + 1}")
    print(f"Best accuracy: {final_accuracy}")

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111)
    # x_labels = [i for i in range(MIN_DEPTH, MAX_DEPTH)]
    # ax.set_xlabel("Tree depth", fontsize=18)
    # ax.set_ylabel("Accuracy in %", rotation=90, fontsize=18)
    # ax.set_xticks(x_labels)

    # tp = ax.bar(x_labels, accuracy_list, color="blue")

    # plt.title("Dokładność modelu dla wybranych głębokości", fontweight='bold', fontsize=26)
    # plt.savefig("Dokładność modelu dla wybranych głębokości")
    # plt.show()

    # print("Done")

    return accuracy_list


def testing_rf(reader: DataReader, solver: RandomForest, target: str):
    print("Fitting ...")
    generated_forest = []
    accuracy_list = []
    for tree_count in TREE_NUMS:
        generated_forest.append(solver.fit(reader.get_train_df(), tree_count))

    for forest in generated_forest:
        accuracy_list.append(round(solver.check_accuracy(reader.get_val_df(), forest, target) * 100, 2))

    best_accuracy = max(accuracy_list)
    best_forest = generated_forest[accuracy_list.index(best_accuracy)]

    final_accuracy = round(solver.check_accuracy(reader.get_test_df(), best_forest, target) * 100, 2)

    print(f"Best tree count: {TREE_NUMS[generated_forest.index(best_forest)]}")
    print(f"Best accuracy: {final_accuracy}")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    x_labels = [i for i in range(1, len(TREE_NUMS) + 1)]
    ax.set_xlabel("Trees number in forest", fontsize=18)
    ax.set_ylabel("Accuracy in %", rotation=90, fontsize=18)
    ax.set_xticks(x_labels)

    tp = ax.bar(x_labels, accuracy_list, color="blue")

    plt.title("Dokładność RF dla wybranej ilości drzew", fontweight='bold', fontsize=26)
    plt.savefig("Dokładność RF dla wybranej ilości drzew")
    plt.show()

    print("Done")


if __name__ == "__main__":
    print("Input csv file with data:")
    data_set = str(input())
    print("Input file to save testing results from id3:")
    id3_file = str(input())
    id3_result_list = []
    with open(data_set) as file_h:
        reader = DataReader()
        id3_solver = DecisionSolver()
        rf_solver = RandomForest()
        for rand_seed in SETS_RAND_SEED:
            reader.read_data(file_h, rand_seed)
            target = reader.get_train_df().columns.tolist()[-1]
            id3_result_list.append(testing_id3(reader, id3_solver, target)) 
            # testing_rf(reader, rf_solver, target)

    with open(id3_file, "w") as file_h:
        for row in range(len(id3_result_list)):
            for column in range(MAX_DEPTH - 1):
                file_h.write(f"{id3_result_list[row][column]}")
            file_h.write('\n')