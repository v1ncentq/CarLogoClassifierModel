from pathlib import Path
import yaml
import torch

from src.data.loader import load_samples_from_csv, build_dataloader
from src.data.dataset import CarLogoDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.cnn_model import CarLogoCNN
from src.models.resnet_gray import ResNetGray
from src.training.trainer import Trainer
from src.training.losses import get_loss_function
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.utils.device import get_device
from src.utils.file_manager import ensure_dir, save_json
from src.utils.seed import set_seed


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_datasets_and_loaders(data_config: dict):
    splits_dir = Path(data_config["splits_dir"])

    train_samples = load_samples_from_csv(str(splits_dir / "train.csv"))
    val_samples = load_samples_from_csv(str(splits_dir / "val.csv"))

    image_size = tuple(data_config["image_size"])

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = CarLogoDataset(train_samples, transform=train_transform)
    val_dataset = CarLogoDataset(val_samples, transform=val_transform)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"]
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"]
    )

    return train_loader, val_loader, train_dataset, val_dataset


def build_model(model_config: dict) -> torch.nn.Module:
    model_name = model_config["model_name"].lower()

    if model_name == "car_logo_cnn":
        return CarLogoCNN(
            num_classes=model_config["num_classes"],
            input_channels=model_config["input_channels"],
            base_filters=model_config["base_filters"],
            dropout_rate=model_config["dropout_rate"]
        )

    if model_name == "resnet18_gray":
        return ResNetGray(
            num_classes=model_config["num_classes"],
            pretrained=model_config.get("pretrained", False)
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def main() -> None:
    print("\nStarting training process...")

    data_config = load_yaml_config("configs/data_config.yaml")
    model_config = load_yaml_config("configs/model_config.yaml")
    train_config = load_yaml_config("configs/train_config.yaml")

    set_seed(data_config["random_state"])
    
    requested_device = train_config.get("device")
    device = get_device(requested_device)

    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    print("\nLoading datasets and creating data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = build_datasets_and_loaders(data_config)

    print("\nTrain samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Batch size:", data_config["batch_size"])

    print("\nBuilding model...")
    model = build_model(model_config).to(device)

    criterion = get_loss_function(train_config["criterion_name"])
    optimizer = create_optimizer(
        model=model,
        optimizer_name=train_config["optimizer_name"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"]
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=train_config["scheduler_name"],
        step_size=train_config.get("step_size"),
        gamma=train_config.get("gamma")
    )

    save_dir = Path("saved_models/best")
    ensure_dir(str(save_dir))
    save_path = str(save_dir / "best_model.pth")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=train_config["epochs"],
        save_path=save_path,
        scheduler=scheduler
    )

    history = trainer.fit()

    ensure_dir("reports/metrics")
    save_json(history, "reports/metrics/training_history.json")

    print("Training history saved to reports/metrics/training_history.json")

if __name__ == "__main__":
    main()