import torch
from dlutils.models.nd import ConvNd, NormNd
from dlutils.models.resnext import _ResNeXt
from dlutils.models.seresnet import _SELayer


class _SEBottleneckX(torch.nn.Module):
    expansion = 4
    start_filts = 64

    def __init__(self, in_channels: int, channels: int, stride: int,
                 cardinality: int, width: int, n_dim: int, norm_layer: str,
                 reduction: int = 16):
        """
        Squeeze and Excitation ResNeXt Block
        Parameters
        ----------
        in_channels : int
            number of input channels
        stride : int
            stride of 3x3 convolution layer
        cardinality : int
            number of convolution groups
        width : int
            width of resnext block
        n_dim : int
            dimensionality of convolutions
        norm_layer : str
            type of normalization layer
        reduction : int
            reduction for se layer
        """
        super().__init__()
        out_channels = channels * self.expansion
        if cardinality == 1:
            rc = channels
        else:
            width_ratio = channels * (width / self.start_filts)
            rc = cardinality * math.floor(width_ratio)

        self.conv_reduce = ConvNd(n_dim, in_channels, rc, kernel_size=1,
                                  stride=1, padding=0, bias=False)
        self.bn_reduce = NormNd(norm_layer, n_dim, rc)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv_conv = ConvNd(n_dim, rc, rc, kernel_size=3,
                                stride=stride, padding=1,
                                groups=cardinality, bias=False)
        self.bn = NormNd(norm_layer, n_dim, rc)

        self.conv_expand = ConvNd(n_dim, rc, out_channels, kernel_size=1,
                                  stride=1, padding=0, bias=False)
        self.bn_expand = NormNd(norm_layer, n_dim, out_channels)

        self.shortcut = torch.nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut.add_module(
                'shortcut_conv', ConvNd(n_dim, in_channels, out_channels,
                                        kernel_size=1, stride=stride,
                                        padding=0, bias=False))
            self.shortcut.add_module(
                'shortcut_bn', NormNd(norm_layer, n_dim, out_channels))

        self.selayer = _SELayer(n_dim, out_channels, reduction=reduction)

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

        out = self.conv_reduce(x)
        out = self.relu(self.bn_reduce(out))

        out = self.conv_conv(out)
        out = self.relu(self.bn(out))

        out = self.conv_expand(out)
        out = self.bn_expand(out)

        out = self.selayer(out)

        res = self.shortcut(identity)
        return self.relu(res + out)


_SERESNEXT_CONFIGS = {
    "26": {"block": _SEBottleneckX, "layers": [2, 2, 2, 2]},
    "50": {"block": _SEBottleneckX, "layers": [3, 4, 6, 3]},
    "101": {"block": _SEBottleneckX, "layers": [3, 4, 23, 3]},
    "152": {"block": _SEBottleneckX, "layers": [3, 8, 36, 3]},
}


class SEResNeXt26(_ResNeXt):
    def __init__(self, num_classes: int, in_channels: int, cardinality: int,
                 width: int = 4, start_filts: int = 64,
                 start_mode: str = '7x7', n_dim: int = 2,
                 norm_layer: str = 'Batch'):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         cardinality=cardinality, width=width,
                         start_mode=start_mode,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNEXT_CONFIGS['26'])


class SEResNeXt50(_ResNeXt):
    def __init__(self, num_classes: int, in_channels: int, cardinality: int,
                 width: int = 4, start_filts: int = 64,
                 start_mode: str = '7x7', n_dim: int = 2,
                 norm_layer: str = 'Batch'):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         cardinality=cardinality, width=width,
                         start_mode=start_mode,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNEXT_CONFIGS['50'])


class SEResNeXt101(_ResNeXt):
    def __init__(self, num_classes: int, in_channels: int, cardinality: int,
                 width: int = 4, start_filts: int = 64,
                 start_mode: str = '7x7', n_dim: int = 2,
                 norm_layer: str = 'Batch'):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         cardinality=cardinality, width=width,
                         start_mode=start_mode,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNEXT_CONFIGS['101'])


class SEResNeXt152(_ResNeXt):
    def __init__(self, num_classes: int, in_channels: int, cardinality: int,
                 width: int = 4, start_filts: int = 64,
                 start_mode: str = '7x7', n_dim: int = 2,
                 norm_layer: str = 'Batch'):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         cardinality=cardinality, width=width,
                         start_mode=start_mode,
                         norm_layer=norm_layer, n_dim=n_dim,
                         start_filts=start_filts, **_SERESNEXT_CONFIGS['152'])
