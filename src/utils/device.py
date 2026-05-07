import torch


def get_device(requested_device: str | None = None) -> torch.device:
    """
    Повертає коректний пристрій для PyTorch.

    requested_device:
        - "cuda" -> використовувати GPU, якщо доступний
        - "cpu"  -> примусово CPU
        - None   -> автоматичний вибір: CUDA, якщо є, інакше CPU
    """
    if requested_device is not None:
        requested_device = requested_device.lower().strip()

        if requested_device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("CUDA недоступна. Використовується CPU.")
            return torch.device("cpu")

        if requested_device == "cpu":
            return torch.device("cpu")

        raise ValueError(f"Unsupported device: {requested_device}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")