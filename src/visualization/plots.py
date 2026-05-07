from pathlib import Path
import matplotlib.pyplot as plt


def plot_metric_curve(
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | None = None
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values) + 1), values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path is not None:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight")

    plt.show()


def plot_bar_distribution(
    labels: list[str],
    counts: list[int],
    title: str,
    save_path: str | None = None
) -> None:
    plt.figure(figsize=(12, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()

    if save_path is not None:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight")

    plt.show()