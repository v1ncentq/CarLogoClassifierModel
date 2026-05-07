from pathlib import Path
import csv
from torch.utils.data import DataLoader


def load_samples_from_csv(csv_path: str) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = row["image_path"]
            label = int(row["label"])
            samples.append((image_path, label))

    return samples


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int
) -> DataLoader:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader


def load_samples_from_directory(
    root_dir: str,
    allowed_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
) -> list[tuple[str, int]]:
    root_path = Path(root_dir)
    class_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])
    class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}

    samples: list[tuple[str, int]] = []

    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in allowed_extensions:
                samples.append((str(image_path), label))

    return samples