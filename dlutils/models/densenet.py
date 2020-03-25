from collections import OrderedDict
from typing import Sequence

import torch
from torch.nn import functional as F

from dlutils.models.nd import NormNd, ConvNd, PoolingNd


class _DenseLayer(torch.nn.Sequential):
    def __init__(self, num_input_features: int, growth_rate: int,
                 bn_size: int, drop_rate: float,
                 n_dim: int = 2, norm_type: str = "Batch"):
        super().__init__()
        self.add_module('norm1', NormNd(norm_type, n_dim,
                                        num_input_features)),
        self.add_module('relu1', torch.nn.ReLU(inplace=True)),
        self.add_module('conv1', ConvNd(n_dim, num_input_features,
                                        bn_size * growth_rate,
                                        kernel_size=1, stride=1,
                                        bias=False)),
        self.add_module('norm2', NormNd(norm_type, n_dim,
                                        bn_size * growth_rate)),
        self.add_module('relu2', torch.nn.ReLU(inplace=True)),
        self.add_module('conv2', ConvNd(n_dim, bn_size * growth_rate,
                                        growth_rate,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(torch.nn.Sequential):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int,
                 growth_rate: int, drop_rate: float, n_dim: int = 2,
                 norm_type: str = "Batch"):
        super().__init__()

        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, n_dim,
                                norm_type)

            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(torch.nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int,
                 n_dim: int = 2, norm_type: str = "Batch"):
        super().__init__()
        self.add_module('norm', NormNd(norm_type, n_dim,
                                       num_input_features))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', ConvNd(n_dim, num_input_features,
                                       num_output_features,
                                       kernel_size=1, stride=1,
                                       bias=False))
        self.add_module('pool', PoolingNd("AdaptiveAvg", n_dim,
                                          output_size=2))


class DenseNet(torch.nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, num_classes: int, in_channels: int, growth_rate: int = 32,
                 block_config: Sequence[int] = (6, 12, 24, 16),
                 num_init_features: int = 64, bn_size: int = 4,
                 drop_rate: float = 0, n_dim: int = 2, pool_type: str = "Max",
                 norm_type: str = "Batch"):

        super().__init__()

        # First convolution
        self.features = torch.nn.Sequential(OrderedDict([
            ('conv0', ConvNd(n_dim, in_channels, num_init_features,
                             kernel_size=7,
                             stride=2, padding=3, bias=False)),
            ('norm0', NormNd(norm_type, n_dim, num_init_features)),
            ('relu0', torch.nn.ReLU(inplace=True)),
            ('pool0', PoolingNd(pool_type, n_dim, kernel_size=3, stride=2,
                                padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate, n_dim=n_dim,
                                norm_type=norm_type)

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    n_dim=n_dim, norm_type=norm_type)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final norm
        self.features.add_module('norm5', NormNd(norm_type, n_dim,
                                                 num_features))

        self.pool = PoolingNd("AdaptiveAvg", n_dim, 1)

        # Linear layer
        self.classifier = torch.nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, ConvNd):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            elif isinstance(m, NormNd):
                if hasattr(m.norm, "weight") and m.norm.weight is not None:
                    torch.nn.init.constant_(m.norm.weight, 1)

                if hasattr(m.norm, "bias") and m.norm.bias is not None:
                    torch.nn.init.constant_(m.norm.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.pool(out).view(x.size(0), -1)
        out = self.classifier(out)
        return out
