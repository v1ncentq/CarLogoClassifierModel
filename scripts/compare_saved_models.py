from pathlib import Path
import sys
import json
import time
import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_samples_from_csv, build_dataloader
from src.data.dataset import CarLogoDataset
from src.data.transforms import get_val_transforms
from src.models.cnn_model import CarLogoCNN
from src.models.resnet_gray import ResNetGray
from src.training.losses import get_loss_function
from src.evaluation.evaluate import evaluate_model
from src.utils.device import get_device
from src.utils.file_manager import ensure_dir


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model_weights(model: torch.nn.Module, weights_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def build_test_loader(data_config: dict):
    test_csv = Path(data_config["splits_dir"]) / "test.csv"
    test_samples = load_samples_from_csv(str(test_csv))

    test_transform = get_val_transforms(tuple(data_config["image_size"]))

    test_dataset = CarLogoDataset(
        samples=test_samples,
        transform=test_transform
    )

    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"]
    )

    return test_dataset, test_loader


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(model: torch.nn.Module, test_loader, device: torch.device) -> float:
    model.eval()

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            total_time += end_time - start_time
            total_images += images.size(0)

    return total_time / total_images if total_images > 0 else 0.0


def evaluate_saved_model(
    model_name: str,
    model: torch.nn.Module,
    weights_path: str,
    test_loader,
    criterion,
    device: torch.device
) -> dict:
    print("\n" + "=" * 70)
    print(f"Evaluating: {model_name}")
    print(f"Weights: {weights_path}")
    print("=" * 70)

    model = load_model_weights(model, weights_path, device)

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    params = count_parameters(model)
    avg_time = measure_inference_time(model, test_loader, device)

    result = {
        "model": model_name,
        "weights_path": weights_path,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "parameters": params,
        "avg_inference_time_sec": avg_time
    }

    print(f"Loss:      {result['loss']:.4f}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1-score:  {result['f1']:.4f}")
    print(f"Params:    {result['parameters']}")
    print(f"Time/img:  {result['avg_inference_time_sec']:.6f} sec")

    return result


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 110)
    print("MODEL COMPARISON")
    print("=" * 110)

    print(
        f"{'Model':<18} "
        f"{'Loss':<10} "
        f"{'Accuracy':<10} "
        f"{'Precision':<10} "
        f"{'Recall':<10} "
        f"{'F1':<10} "
        f"{'Params':<15} "
        f"{'Time/img':<10}"
    )

    print("-" * 110)

    for r in results:
        print(
            f"{r['model']:<18} "
            f"{r['loss']:<10.4f} "
            f"{r['accuracy']:<10.4f} "
            f"{r['precision']:<10.4f} "
            f"{r['recall']:<10.4f} "
            f"{r['f1']:<10.4f} "
            f"{r['parameters']:<15} "
            f"{r['avg_inference_time_sec']:<10.6f}"
        )

    print("=" * 110)


def main() -> None:
    data_config = load_yaml_config("configs/data_config.yaml")
    train_config = load_yaml_config("configs/train_config.yaml")
    model_config = load_yaml_config("configs/model_config.yaml")

    device = get_device(train_config.get("device"))

    print("Selected device:", device)
    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(0))

    test_dataset, test_loader = build_test_loader(data_config)
    print("Test samples:", len(test_dataset))

    criterion = get_loss_function(train_config["criterion_name"])

    num_classes = model_config["num_classes"]

    cnn_model = CarLogoCNN(
        num_classes=num_classes,
        input_channels=1,
        base_filters=32,
        dropout_rate=0.2
    )

    resnet_model = ResNetGray(
        num_classes=num_classes,
        pretrained=False
    )

    cnn_weights_path = "saved_models/cnn/model_cnn.pth"
    resnet_weights_path = "saved_models/resnet18_gray/model_resnet18_gray.pth"

    results = []

    results.append(
        evaluate_saved_model(
            model_name="Custom CNN",
            model=cnn_model,
            weights_path=cnn_weights_path,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
    )

    results.append(
        evaluate_saved_model(
            model_name="ResNet18 Gray",
            model=resnet_model,
            weights_path=resnet_weights_path,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
    )

    print_table(results)

    ensure_dir("reports/metrics")

    output_path = "reports/metrics/model_comparison.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()