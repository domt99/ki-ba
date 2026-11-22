from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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