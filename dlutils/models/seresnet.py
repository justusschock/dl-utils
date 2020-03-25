from typing import Union, Sequence

import torch

from dlutils.models.nd import ConvNd, PoolingNd, NormNd
from dlutils.models.resnet import _conv1x1, _conv3x3, _ResNet


class _SELayer(torch.nn.Module):
    def __init__(self, n_dim: int, channel: int, reduction: int = 16):
        """
        Squeeze and Excitation Layer
        https://arxiv.org/abs/1709.01507
        Parameters
        ----------
        n_dim : int
            dimensionality of convolution
        channel : int
            number of input channel
        reduction : int
            channel reduction factor
        """
        super().__init__()
        self.pool = PoolingNd('AdaptiveAvg', n_dim, 1)
        self.fc = torch.nn.Sequential(
            ConvNd(n_dim, channel, channel // reduction, kernel_size=1,
                   bias=False),
            torch.nn.ReLU(inplace=True),
            ConvNd(n_dim, channel // reduction, channel, kernel_size=1,
                   bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input through layer
        Parameters
        ----------
        x : torch.Tensor
            input
        Returns
        -------
        torch.Tensor
            output
        """
        y = self.pool(x)
        y = self.fc(y)
        return x * y


class _SEBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int,
                 stride: Union[int, Sequence[int]] = 1,
                 downsample: torch.nn.Module = None,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 reduction: int = 16):
        """
        Squeeze and Excitation Basic ResNet block
        Parameters
        ----------
        inplanes : int
            number of input channels
        planes : int
            number of intermediate channels
        stride : int or tuple
            stride of first convolution
        downsample : nn.Module
            downsampling in residual path
        norm_layer : str
            type of normalisation layer
        n_dim : int
            dimensionality of convolution
        reduction : int
            reduction for squeeze and excitation layer
        """
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = _conv3x3(inplanes, planes, stride, n_dim=n_dim)
        self.bn1 = NormNd(norm_layer, n_dim, planes)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = _conv3x3(planes, planes, n_dim=n_dim)
        self.bn2 = NormNd(norm_layer, n_dim, planes)

        self.downsample = downsample
        self.stride = stride

        self.selayer = _SELayer(n_dim, planes * self.expansion,
                                reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input through block
        Parameters
        ----------
        x : torch.Tensor
            input
        Returns
        -------
        torch.Tensor
            output
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.selayer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _SEBottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int,
                 stride: Union[int, Sequence[int]] = 1,
                 downsample: torch.nn.Module = None,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 reduction: int = 16):
        """
        Squeeze and Excitation Bottleneck ResNet block
        Parameters
        ----------
        inplanes : int
            number of input channels
        planes : int
            number of intermediate channels
        stride : int or tuple
            stride of first convolution
        downsample : nn.Module
            downsampling in residual path
        norm_layer : str
            type of normalisation layer
        n_dim : int
            dimensionality of convolution
        reduction : int
            reduction for squeeze and excitation layer
        """
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1(inplanes, planes, n_dim=n_dim)
        self.bn1 = NormNd(norm_layer, n_dim, planes)
        self.conv2 = _conv3x3(planes, planes, stride, n_dim=n_dim)
        self.bn2 = NormNd(norm_layer, n_dim, planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion, n_dim=n_dim)
        self.bn3 = NormNd(norm_layer, n_dim, planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.selayer = _SELayer(n_dim, planes * self.expansion,
                                reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input through block
        Parameters
        ----------
        x : torch.Tensor
            input
        Returns
        -------
        torch.Tensor
            output
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


_SERESNET_CONFIGS = {
    "18": {"block": _SEBasicBlock, "layers": [2, 2, 2, 2]},
    "26": {"block": _SEBottleneck, "layers": [2, 2, 2, 2]},
    "34": {"block": _SEBasicBlock, "layers": [3, 4, 6, 3]},
    "50": {"block": _SEBottleneck, "layers": [3, 4, 6, 3]},
    "101": {"block": _SEBottleneck, "layers": [3, 4, 23, 3]},
    "152": {"block": _SEBottleneck, "layers": [3, 8, 36, 3]},
}


class SEResNet18(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNET_CONFIGS['18'])


class SEResNet26(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNET_CONFIGS['26'])


class SEResNet34(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNET_CONFIGS['34'])


class SEResNet50(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNET_CONFIGS['50'])


class SEResNet101(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNET_CONFIGS['101'])


class SEResNet152(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNET_CONFIGS['152'])
