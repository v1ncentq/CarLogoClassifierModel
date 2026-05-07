from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_history(history: dict[str, list[float]], save_dir: str | None = None) -> None:
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_accuracy"]
    val_acc = history["val_accuracy"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_dir is not None:
        output_path = Path(save_dir) / "loss_curve.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    if save_dir is not None:
        output_path = Path(save_dir) / "accuracy_curve.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()