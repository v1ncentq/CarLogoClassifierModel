from pathlib import Path
import argparse
import yaml
import torch

from src.models.cnn_model import CarLogoCNN
from src.models.resnet_gray import ResNetGray
from src.data.transforms import get_inference_transforms
from src.inference.predictor import Predictor
from src.utils.device import get_device


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions")
    return parser.parse_args(argv)


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def main(argv=None) -> None:
    args = parse_args(argv)

    print("\nLoading configs for inference...")

    inference_config = load_yaml_config("configs/inference_config.yaml")
    model_config = load_yaml_config("configs/model_config.yaml")

    requested_device = inference_config.get("device")
    device = get_device(requested_device)

    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    weights_path = args.weights or inference_config["weights_path"]
    image_path = args.image
    top_k = args.top_k

    class_names = inference_config["class_names"]

    transform = get_inference_transforms(tuple(inference_config["image_size"]))

    model = build_model(model_config)

    predictor = Predictor(
        model=model,
        weights_path=weights_path,
        transform=transform,
        class_names=class_names,
        device=device
    )

    print("\nRunning prediction...")
    result = predictor.predict_top_k(image_path=image_path, k=top_k)

    print("\nPrediction result")
    print("=" * 50)
    print("Image:", image_path)
    print("Predicted class:", result["predicted_class"])
    print(f"Confidence: {result['confidence']:.4f}")

    if result["top_k"] is not None:
        print("\nTop predictions:")
        for idx, (class_name, score) in enumerate(result["top_k"], start=1):
            print(f"{idx}. {class_name}: {score:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()