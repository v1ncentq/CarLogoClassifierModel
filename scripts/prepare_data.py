from pathlib import Path
import yaml
from tqdm import tqdm

from src.data.loader import load_samples_from_directory
from src.data.preprocess import process_and_save_image
from src.data.split import split_dataset, save_split_to_csv
from src.utils.file_manager import ensure_dir
from src.utils.seed import set_seed


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_processed_images(config: dict) -> None:
    raw_dir = Path(config["raw_data_dir"])
    processed_dir = Path(config["processed_data_dir"])
    image_size = tuple(config["image_size"])
    grayscale = config["grayscale"]

    ensure_dir(str(processed_dir))

    samples = load_samples_from_directory(str(raw_dir))

    for image_path, label in tqdm(samples, desc="Processing images"):
        input_path = Path(image_path)
        class_name = input_path.parent.name
        output_class_dir = processed_dir / class_name
        ensure_dir(str(output_class_dir))

        output_path = output_class_dir / input_path.name

        process_and_save_image(
            input_path=str(input_path),
            output_path=str(output_path),
            image_size=image_size,
            grayscale=grayscale
        )


def build_and_save_splits(config: dict) -> None:
    processed_dir = Path(config["processed_data_dir"])
    splits_dir = Path(config["splits_dir"])
    ensure_dir(str(splits_dir))

    samples = load_samples_from_directory(str(processed_dir))

    train_samples, val_samples, test_samples = split_dataset(
        samples=samples,
        train_ratio=config["train_split"],
        val_ratio=config["val_split"],
        test_ratio=config["test_split"],
        random_state=config["random_state"]
    )

    save_split_to_csv(train_samples, str(splits_dir / "train.csv"))
    save_split_to_csv(val_samples, str(splits_dir / "val.csv"))
    save_split_to_csv(test_samples, str(splits_dir / "test.csv"))


def main() -> None:
    config_path = Path("configs/data_config.yaml")
    config = load_yaml_config(str(config_path))

    set_seed(config["random_state"])

    prepare_processed_images(config)
    build_and_save_splits(config)


if __name__ == "__main__":
    main()