from pathlib import Path
import yaml
import torch

from src.data.loader import load_samples_from_csv, build_dataloader
from src.data.dataset import CarLogoDataset
from src.data.transforms import get_val_transforms
from src.models.cnn_model import CarLogoCNN
from src.models.resnet_gray import ResNetGray
from src.training.losses import get_loss_function
from src.evaluation.evaluate import evaluate_model, collect_predictions
from src.evaluation.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix
from src.utils.device import get_device
from src.utils.file_manager import ensure_dir, save_json


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_test_loader(data_config: dict):
    splits_dir = Path(data_config["splits_dir"])
    test_samples = load_samples_from_csv(str(splits_dir / "test.csv"))

    test_transform = get_val_transforms(tuple(data_config["image_size"]))
    test_dataset = CarLogoDataset(test_samples, transform=test_transform)

    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"]
    )

    return test_loader, test_dataset


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
    print("\nLoading configs for evaluation...")
    
    data_config = load_yaml_config("configs/data_config.yaml")
    model_config = load_yaml_config("configs/model_config.yaml")
    train_config = load_yaml_config("configs/train_config.yaml")
    inference_config = load_yaml_config("configs/inference_config.yaml")

    requested_device = train_config.get("device")
    device = get_device(requested_device)

    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
    test_loader, test_dataset = build_test_loader(data_config)
    print(f"Test samples: {len(test_dataset)}")

    model = build_model(model_config).to(device)

    weights_path = inference_config["weights_path"]
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    criterion = get_loss_function(train_config["criterion_name"])

    print("\nRunning evaluation...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    y_true, y_pred = collect_predictions(
        model=model,
        test_loader=test_loader,
        device=device
    )

    cm = compute_confusion_matrix(y_true, y_pred)

    class_names = inference_config["class_names"]

    ensure_dir("reports/figures")
    ensure_dir("reports/metrics")

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path="reports/figures/confusion_matrix.png"
    )

    save_json(metrics, "reports/metrics/test_metrics.json")

    print("\nEvaluation results:")
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print("\nSaved:")
    print("- reports/metrics/test_metrics.json")
    print("- reports/figures/confusion_matrix.png")


if __name__ == "__main__":
    main()