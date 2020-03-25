from typing import Sequence, Union

import torch

from dlutils.models.nd import ConvNd, PoolingNd, NormNd


class _VGG(torch.nn.Module):
    def __init__(self, feature_cfg: Sequence[Union[int, str]], num_classes: int,
                 in_channels: int, n_dim: int = 2, norm_type: str = "Batch",
                 pool_type="Max"):
        super().__init__()

        self.features = self.make_layers(feature_cfg,
                                         in_channels=in_channels,
                                         norm_type=norm_type,
                                         n_dim=n_dim,
                                         pool_type=pool_type)

        self.avgpool = PoolingNd("AdaptiveAvg", n_dim, 7)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * pow(7, n_dim), 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )
        self.reset_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, ConvNd):
                torch.nn.init.kaiming_normal_(m.conv.weight,
                                              mode='fan_out',
                                              nonlinearity='relu')
                if m.conv.bias is not None:
                    torch.nn.init.constant_(m.conv.bias, 0)
            elif isinstance(m, NormNd):
                if hasattr(m.norm, "weight") and m.norm.weight is not None:
                    torch.nn.init.constant_(m.norm.weight, 1)

                if hasattr(m.norm, "bias") and m.norm.bias is not None:
                    torch.nn.init.constant_(m.norm.bias, 0)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg: Sequence[Union[int, str]], in_channels: int,
                    norm_type: str = None, n_dim: int = 2,
                    pool_type: str = "Max") -> torch.nn.Sequential:
        layers = []

        for v in cfg:
            if v == 'P':
                layers += [PoolingNd(pool_type, n_dim, kernel_size=2,
                                     stride=2)]
            else:
                _layers = [ConvNd(n_dim, in_channels, v, kernel_size=3,
                                  padding=1)]
                if norm_type is not None:
                    _layers.append(NormNd(norm_type, n_dim, v))

                _layers.append(torch.nn.ReLU(inplace=True))
                layers += _layers
                in_channels = v

        return torch.nn.Sequential(*layers)


_VGG_CONFIGS = {
    '11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    '13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512,
           512, 'P'],
    '16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512,
           'P', 512, 512, 512, 'P'],
    '19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512,
           512, 512, 'P', 512, 512, 512, 512, 'P'],
}


class VGG11(_VGG):
    def __init__(self, num_classes: int,
                 in_channels: int, n_dim: int = 2, norm_type: str = "Batch",
                 pool_type="Max"):
        feature_cfg = _VGG_CONFIGS['11']
        super().__init__(feature_cfg=feature_cfg, num_classes=num_classes,
                         in_channels=in_channels, n_dim=n_dim,
                         norm_type=norm_type, pool_type=pool_type)


class VGG13(_VGG):
    def __init__(self, num_classes: int,
                 in_channels: int, n_dim: int = 2, norm_type: str = "Batch",
                 pool_type="Max"):
        feature_cfg = _VGG_CONFIGS['13']
        super().__init__(feature_cfg=feature_cfg, num_classes=num_classes,
                         in_channels=in_channels, n_dim=n_dim,
                         norm_type=norm_type, pool_type=pool_type)


class VGG16(_VGG):
    def __init__(self, num_classes: int,
                 in_channels: int, n_dim: int = 2, norm_type: str = "Batch",
                 pool_type="Max"):
        feature_cfg = _VGG_CONFIGS['16']
        super().__init__(feature_cfg=feature_cfg, num_classes=num_classes,
                         in_channels=in_channels, n_dim=n_dim,
                         norm_type=norm_type, pool_type=pool_type)


class VGG19(_VGG):
    def __init__(self, num_classes: int,
                 in_channels: int, n_dim: int = 2, norm_type: str = "Batch",
                 pool_type="Max"):
        feature_cfg = _VGG_CONFIGS['19']
        super().__init__(feature_cfg=feature_cfg, num_classes=num_classes,
                         in_channels=in_channels, n_dim=n_dim,
                         norm_type=norm_type, pool_type=pool_type)
