import torch.nn as nn


def get_loss_function(name: str):
    loss_name = name.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported loss function: {name}")