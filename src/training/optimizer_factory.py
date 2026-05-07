import torch.optim as optim


def create_optimizer(
    model,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0
):
    name = optimizer_name.lower()

    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(
    optimizer,
    scheduler_name: str | None,
    step_size: int | None = None,
    gamma: float | None = None
):
    if scheduler_name is None:
        return None

    name = scheduler_name.lower()

    if name == "step_lr":
        if step_size is None or gamma is None:
            raise ValueError("step_lr requires step_size and gamma")
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

    if name == "exponential_lr":
        if gamma is None:
            raise ValueError("exponential_lr requires gamma")
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")