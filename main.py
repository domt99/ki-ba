import argparse
import time

from ki_ba.framework import Framework
from ki_ba.plots import plot_confusion_matrix, plot_learning_curves, plot_misclassified
from ki_ba.results import create_run_folder, get_initials, save_result


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MLP and save the results")
    parser.add_argument(
        "--template", help="template file in configs/, overrides TEMPLATE in .env"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="also evaluate the test data (only once, for the final configuration)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    initials = get_initials()
    framework = Framework(template=args.template)

    start = time.perf_counter()
    epoch_errors, train_accuracy, validation_accuracy = framework.train()
    duration_seconds = time.perf_counter() - start

    folder = create_run_folder(framework.config, initials)
    topology = "-".join(str(n) for n in framework.config.topology)
    plot_path = plot_learning_curves(
        epoch_errors, train_accuracy, validation_accuracy, folder / "learning_curves.png",
        best_epoch=framework.best_epoch,
        title=f"Learning curves ({topology}, {framework.config.optimizer_type})",
    )
    print(f"Saved plot: {plot_path}")

    test_accuracy = None
    if args.test:
        test_accuracy = evaluate_test_data(framework, folder)

    result_path = save_result(folder, framework, initials, duration_seconds, test_accuracy)
    print(f"Saved result: {result_path}")


def evaluate_test_data(framework, folder):
    test_images, test_labels = framework.data_prep.get_test_data()
    if test_images is None:
        print("Test accuracy: no test data")
        return None

    test_accuracy = framework.calculate_accuracy(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy:.2%}")
    confusion_matrix = framework.calculate_confusion_matrix(test_images, test_labels)
    print_confusion_matrix(confusion_matrix)
    plot_path = plot_confusion_matrix(confusion_matrix, folder / "confusion_matrix.png")
    print(f"Saved plot: {plot_path}")

    indices, true_classes, predicted_classes, confidences = framework.find_misclassified(
        test_images, test_labels
    )
    plot_path = plot_misclassified(
        test_images, indices, true_classes, predicted_classes, confidences,
        folder / "misclassified.png",
    )
    print(f"Saved plot: {plot_path}")
    return test_accuracy


if __name__ == "__main__":
    main()
