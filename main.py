from ki_ba.framework import Framework
from ki_ba.plots import plot_confusion_matrix

def print_confusion_matrix(confusion_matrix):
    class_count = len(confusion_matrix)
    width = max(len(str(confusion_matrix.max())), 4)

    print()
    print("Confusion matrix (rows: true, columns: predicted)")
    header = " " * 6 + "".join(f"{c:>{width + 1}}" for c in range(class_count)) + " │ correct"
    print(header)
    print("─" * len(header))
    for true_class, row in enumerate(confusion_matrix):
        cells = "".join(f"{count:>{width + 1}}" for count in row)
        row_accuracy = row[true_class] / row.sum() if row.sum() > 0 else 0
        print(f"{true_class:>4}  {cells} │ {row_accuracy:7.2%}")


def main():
    framework = Framework()

    framework.train()
    test_images, test_labels = framework.data_prep.get_test_data()
    if test_images is not None:
        test_accuracy = framework.calculate_accuracy(test_images, test_labels)
        print(f"Test accuracy: {test_accuracy:.2%}")
        confusion_matrix = framework.calculate_confusion_matrix(test_images, test_labels)
        print_confusion_matrix(confusion_matrix)
        plot_path = plot_confusion_matrix(confusion_matrix, "plots/confusion_matrix.png")
        print(f"Saved plot: {plot_path}")
    else:
        print("Test accuracy: no test data")


if __name__ == "__main__":
    main()