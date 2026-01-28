from data_reader import DataReader
from naive_bayes_solver import NaiveBayesSolver
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def print_top_words(model: NaiveBayesSolver, vocabulary: list, num=10):
    print(f"\n{'=' * 20} TOP {num} NAJWAŻNIEJSZYCH SŁÓW {'=' * 20}")

    for label in model.classes:
        probs = model.likelihoods[label]

        sorted_indices = np.argsort(probs)
        top_indices = sorted_indices[-num:][::-1]

        top_words = []
        for i in top_indices:
            word = vocabulary[i]
            probability = probs[i]
            top_words.append(f"{word}, p-ństwo: ({probability:.4f})")

        print(f"\nKlasa: {label.upper()}")
        for i, word_info in enumerate(top_words, 1):
            print(f"{i}. {word_info}")


def main():
    seeds = [30, 37, 45, 56, 68]
    alphas = [0.6, 0.7, 0.8, 0.9, 1.0]

    results_val = []
    results_test = []

    reader = DataReader()
    print("======= Podaj plik z danymi do obróbki: =======")
    file = input("Plik (.csv): ")

    # print("======= Podaj random seed: =======")
    # rand_seed = int(input("Random seed: "))

    print("======= Czy używamy bigramów?: =======")
    use_bigrams_input = input("Y/n: ")

    use_bigrams = True if use_bigrams_input == 'Y' else False

    # acc_list_val = []
    # acc_list_test = []

    for alpha in alphas:
        print(f"\n{'=' * 30} Używany alfa: {alpha} {'=' * 30}")
        for seed in seeds:
            print(f"{'=' * 20} Używany seed: {seed} {'=' * 20}")

            reader.read_data(file, seed, use_bigrams)

            x_train = reader.get_train_x()
            y_train = reader.get_train_y()

            x_val = reader.get_val_x()
            y_val = reader.get_val_y()

            x_test = reader.get_test_x()
            y_test = reader.get_test_y()

            model = NaiveBayesSolver(alpha)
            model.fit(x_train, y_train)
            print_top_words(model, reader.vocab)

            predictions_val = model.predict(x_val)
            acc_val = accuracy_score(y_val, predictions_val)
            results_val.append({
                'seed': seed,
                'alpha': alpha,
                'accuracy_val': acc_val
            })

            predictions_test = model.predict(x_test)
            acc_test = accuracy_score(y_test, predictions_test)
            results_test.append({
                'seed': seed,
                'alpha': alpha,
                'accuracy_val': acc_test
            })

            # print("Dokładnosc dla zbioru walidacyjnego: ", accuracy_score(y_val, predictions_val))
            # acc_list_val.append(accuracy_score(y_val, predictions_val))
            # print(classification_report(y_val, predictions_val))

            # print("Dokładnosc dla zbioru testowego: ", accuracy_score(y_test, predictions_test))
            # acc_list_test.append(accuracy_score(y_test, predictions_test))
            # print(classification_report(y_test, predictions_test))

    df_results_val = pd.DataFrame(results_val)
    print("\n=== Średnie wyniki dla Alfa dla zbioru walidacyjnego ===")
    print(df_results_val.groupby('alpha')['accuracy_val'].mean())

    # df_results_test = pd.DataFrame(results_test)
    # print("\n=== Średnie wyniki dla Alfa dla zbioru testowego ===")
    # print(df_results_test.groupby('alpha')['accuracy_val'].mean())

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results_val, x='alpha', y='accuracy_val', marker='o')
    plt.title(f'Wpływ parametru Alpha na dokładność (Bigrams={use_bigrams})')
    plt.xlabel('Alpha')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
