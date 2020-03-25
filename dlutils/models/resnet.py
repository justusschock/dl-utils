import copy
from typing import Union, Sequence

import torch
from dlutils.models.nd import ConvNd, NormNd, PoolingNd

_RESNET_CONFIGS = {
    "18": {"block": _BasicBlock, "layers": [2, 2, 2, 2]},
    "34": {"block": _BasicBlock, "layers": [3, 4, 6, 3]},
    "26": {"block": _Bottleneck, "layers": [2, 2, 2, 2]},
    "50": {"block": _Bottleneck, "layers": [3, 4, 6, 3]},
    "101": {"block": _Bottleneck, "layers": [3, 4, 23, 3]},
    "152": {"block": _Bottleneck, "layers": [3, 8, 36, 3]},
}


def _conv3x3(in_planes: int, out_planes: int,
             stride: Union[int, Sequence[int]] = 1, n_dim: int = 2) -> ConvNd:
    """3x3 convolution with padding"""
    return ConvNd(n_dim, in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int,
             stride: Union[int, Sequence[int]] = 1, n_dim: int = 2) -> ConvNd:
    """1x1 convolution"""
    return ConvNd(n_dim, in_planes, out_planes, kernel_size=1,
                  stride=stride, bias=False)


class _BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int,
                 stride: Union[int, Sequence[int]] = 1,
                 downsample: torch.nn.Module = None,
                 norm_layer: str = "Batch", n_dim: int = 2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int,
                 stride: Union[int, Sequence[int]] = 1,
                 downsample: torch.nn.Module = None,
                 norm_layer: str = "Batch", n_dim: int = 2):
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1(inplanes, planes, n_dim=n_dim)
        self.bn1 = NormNd(norm_layer, n_dim, planes)
        self.conv2 = _conv3x3(planes, planes, stride, n_dim=n_dim)
        self.bn2 = NormNd(norm_layer, n_dim, planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion, n_dim=n_dim)
        self.bn3 = NormNd(norm_layer, n_dim, planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(torch.nn.Module):
    def __init__(self, block: torch.nn.Module, layers: Sequence[int],
                 num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__()

        self.start_filts = start_filts
        self.inplanes = copy.deepcopy(start_filts)
        self.conv1 = ConvNd(n_dim, in_channels, self.inplanes,
                            kernel_size=7, stride=2, padding=3,
                            bias=False)

        self.bn1 = NormNd(norm_layer, n_dim, self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = PoolingNd("Max", n_dim=n_dim, kernel_size=3,
                                 stride=2, padding=1)

        num_layers = 0
        for idx, _layers in enumerate(layers):
            stride = 1 if idx == 0 else 2
            planes = min(self.start_filts * pow(2, idx), self.start_filts * 8)
            _local_layer = self._make_layer(block, planes, _layers,
                                            stride=stride,
                                            norm_layer=norm_layer,
                                            n_dim=n_dim)

            setattr(self, "layer%d" % (idx + 1), _local_layer)
            num_layers += 1

        self.num_layers = num_layers

        self.avgpool = PoolingNd("AdaptiveAvg", n_dim, 1)
        self.fc = torch.nn.Linear(self.inplanes, num_classes)
        self.reset_weights(zero_init_residual=zero_init_residual)

    def reset_weights(self, zero_init_residual: bool = False):

        for m in self.modules():
            if isinstance(m, ConvNd):
                torch.nn.init.kaiming_normal_(m.conv.weight,
                                              mode='fan_out',
                                              nonlinearity='relu')

            elif isinstance(m, NormNd):
                if hasattr(m.norm, "weight") and m.norm.weight is not None:
                    torch.nn.init.constant_(m.norm.weight, 1)
                if hasattr(m.norm, "bias") and m.norm.bias is not None:
                    torch.nn.init.constant_(m.norm.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, _Bottleneck):
                    torch.nn.init.constant_(m.bn3.norm.weight, 0)
                elif isinstance(m, _BasicBlock):
                    torch.nn.init.constant_(m.bn2.norm.weight, 0)

    def _make_layer(self, block: torch.nn.Module, planes: int, blocks: int,
                    stride: Union[int, Sequence[int]] = 1,
                    norm_layer: str = "Batch",
                    n_dim: int = 2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride,
                         n_dim=n_dim),
                NormNd(norm_layer, n_dim, planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        norm_layer, n_dim=n_dim)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, n_dim=n_dim))

        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(self.num_layers):
            x = getattr(self, "layer%d" % (i + 1))(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


_RESNET_CONFIGS = {
    "18": {"block": _BasicBlock, "layers": [2, 2, 2, 2]},
    "26": {"block": _Bottleneck, "layers": [2, 2, 2, 2]},
    "34": {"block": _BasicBlock, "layers": [3, 4, 6, 3]},
    "50": {"block": _Bottleneck, "layers": [3, 4, 6, 3]},
    "101": {"block": _Bottleneck, "layers": [3, 4, 23, 3]},
    "152": {"block": _Bottleneck, "layers": [3, 8, 36, 3]},
}


class ResNet18(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_RESNET_CONFIGS['18'])


class ResNet26(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_RESNET_CONFIGS['26'])


class ResNet34(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_RESNET_CONFIGS['34'])


class ResNet50(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_RESNET_CONFIGS['50'])


class ResNet101(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_RESNET_CONFIGS['101'])


class ResNet152(_ResNet):
    def __init__(self, num_classes: int, in_channels: int,
                 zero_init_residual: bool = False,
                 norm_layer: str = "Batch", n_dim: int = 2,
                 start_filts: int = 64):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         zero_init_residual=zero_init_residual,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_RESNET_CONFIGS['152'])
