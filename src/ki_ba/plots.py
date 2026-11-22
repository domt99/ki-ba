from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

# one-hue sequential ramp (light -> dark blue) for the error counts
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "errors", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIAGONAL_COLOR = "#e8e7e3"
TEXT_DARK = "#1f1f1e"
TEXT_LIGHT = "#ffffff"
TEXT_MUTED = "#8a8985"


def plot_confusion_matrix(confusion_matrix, path, title="Confusion matrix (test data)"):
    class_count = len(confusion_matrix)
    errors = confusion_matrix.astype(float)
    np.fill_diagonal(errors, 0)
    max_error = max(errors.max(), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(errors, cmap=ERROR_CMAP, vmin=0, vmax=max_error)

    for true_class in range(class_count):
        for predicted_class in range(class_count):
            count = confusion_matrix[true_class, predicted_class]
            if true_class == predicted_class:
                ax.add_patch(plt.Rectangle(
                    (predicted_class - 0.5, true_class - 0.5), 1, 1,
                    facecolor=DIAGONAL_COLOR, edgecolor="none",
                ))
                ax.text(predicted_class, true_class, count, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=TEXT_DARK)
            elif count > 0:
                dark_cell = count / max_error > 0.55
                ax.text(predicted_class, true_class, count, ha="center", va="center",
                        fontsize=8, color=TEXT_LIGHT if dark_cell else TEXT_DARK)
            else:
                ax.text(predicted_class, true_class, "·", ha="center", va="center",
                        fontsize=8, color=TEXT_MUTED)

    ax.set_xticks(range(class_count))
    ax.set_yticks(range(class_count))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Misclassifications (diagonal = correct, grey)")
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_misclassified(
    images, indices, true_classes, predicted_classes, confidences, path,
    count=18, columns=6, title="Misclassified test images (most confident first)",
):
    count = min(count, len(indices))
    if count == 0:
        return None

    rows = int(np.ceil(count / columns))
    side = int(np.sqrt(images.shape[1]))

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.9))
    axes = np.atleast_1d(axes).ravel()

    for slot, ax in enumerate(axes):
        ax.axis("off")
        if slot >= count:
            continue
        image = images[indices[slot]].reshape(side, side)
        ax.imshow(image, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(
            f"true {true_classes[slot]} → {predicted_classes[slot]}\n"
            f"{confidences[slot]:.0%} sure",
            fontsize=8, color=TEXT_DARK,
        )

    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


TRAIN_COLOR = "#2a78d6"
VALIDATION_COLOR = "#eb6834"
GRID_COLOR = "#e8e7e3"


def plot_learning_curves(
    epoch_errors, train_accuracy, validation_accuracy, path, best_epoch=None,
    title="Learning curves",
):
    epochs = np.arange(1, len(epoch_errors) + 1)
    has_validation = any(value is not None for value in validation_accuracy)

    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    loss_ax.plot(epochs, epoch_errors, color=TRAIN_COLOR, linewidth=2)
    loss_ax.set_title("Training loss", loc="left", fontsize=10, color=TEXT_DARK)

    accuracy_ax.plot(epochs, train_accuracy, color=TRAIN_COLOR, linewidth=2, label="train")
    _label_line_end(accuracy_ax, epochs[-1], train_accuracy[-1], "train")
    if has_validation:
        accuracy_ax.plot(
            epochs, validation_accuracy, color=VALIDATION_COLOR, linewidth=2, label="validation"
        )
        _label_line_end(accuracy_ax, epochs[-1], validation_accuracy[-1], "val")
    if best_epoch is not None:
        accuracy_ax.axvline(
            best_epoch, color=TEXT_MUTED, linewidth=1, linestyle="--",
            label=f"best val (epoch {best_epoch})",
        )
        accuracy_ax.legend(frameon=False, loc="best", fontsize=9)
    accuracy_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    accuracy_ax.set_title("Accuracy", loc="left", fontsize=10, color=TEXT_DARK)

    if best_epoch is not None:
        loss_ax.axvline(best_epoch, color=TEXT_MUTED, linewidth=1, linestyle="--")

    for ax in (loss_ax, accuracy_ax):
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
        ax.tick_params(length=0, colors=TEXT_MUTED, labelcolor=TEXT_DARK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(TEXT_MUTED)

    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _label_line_end(ax, epoch, value, name):
    ax.annotate(
        f"{name} {value:.2%}", xy=(epoch, value), xytext=(6, 0), textcoords="offset points",
        va="center", fontsize=8, color=TEXT_DARK, annotation_clip=False,
    )
