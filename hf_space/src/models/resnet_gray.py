import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetGray(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        old_conv = self.model.conv1

        kernel_size = (old_conv.kernel_size[0], old_conv.kernel_size[1])
        stride = (old_conv.stride[0], old_conv.stride[1])

        if isinstance(old_conv.padding, tuple):
            padding = (old_conv.padding[0], old_conv.padding[1])
        else:
            padding = old_conv.padding

        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        if pretrained:
            with torch.no_grad():
                self.model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)