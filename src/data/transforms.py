from torchvision import transforms


def get_train_transforms(image_size: tuple[int, int]):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05)
        ),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_val_transforms(image_size: tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_inference_transforms(image_size: tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])