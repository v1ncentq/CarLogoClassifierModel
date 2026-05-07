from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    cm,
    class_names: list[str],
    save_path: str | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight")

    plt.close(fig)