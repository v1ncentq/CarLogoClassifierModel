import argparse
from pathlib import Path

from scripts.prepare_data import main as prepare_data_main
from scripts.train_model import main as train_model_main
from scripts.evaluate_model import main as evaluate_model_main
from scripts.run_inference import main as run_inference_main


DEFAULT_CONFIG_DIR = "configs"
AVAILABLE_MODES = ("prepare_data", "train", "evaluate", "predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Car logo recognition system entry point"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=AVAILABLE_MODES,
        help="Execution mode"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=DEFAULT_CONFIG_DIR,
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for prediction"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cpu or cuda"
    )

    return parser.parse_args()


def run_prepare_data(args: argparse.Namespace) -> None:
    prepare_data_main()


def run_train(args: argparse.Namespace) -> None:
    train_model_main()


def run_evaluate(args: argparse.Namespace) -> None:
    evaluate_model_main()


def run_predict(args) -> None:
    if args.image is None:
        raise ValueError("Prediction mode requires --image argument")

    cli_args = ["--image", args.image]

    if args.weights is not None:
        cli_args.extend(["--weights", args.weights])

    if args.top_k is not None:
        cli_args.extend(["--top_k", str(args.top_k)])

    run_inference_main(cli_args)


def main() -> None:
    args = parse_args()

    if args.mode == "prepare_data":
        run_prepare_data(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "predict":
        run_predict(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()