import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CarLogoCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        base_filters: int = 32,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(input_channels, base_filters),          # 32
            ConvBlock(base_filters, base_filters * 2),        # 64
            ConvBlock(base_filters * 2, base_filters * 4),    # 128
            ConvBlock(base_filters * 4, base_filters * 8)     # 256
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x