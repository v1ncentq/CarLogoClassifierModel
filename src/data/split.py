from pathlib import Path
import csv
import random


def split_dataset(
    samples: list[tuple[str, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must be equal to 1.0")

    shuffled_samples = samples.copy()
    rng = random.Random(random_state)
    rng.shuffle(shuffled_samples)

    n_total = len(shuffled_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = shuffled_samples[:n_train]
    val_samples = shuffled_samples[n_train:n_train + n_val]
    test_samples = shuffled_samples[n_train + n_val:]

    return train_samples, val_samples, test_samples


def save_split_to_csv(samples: list[tuple[str, int]], output_csv: str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])

        for image_path, label in samples:
            writer.writerow([image_path, label])