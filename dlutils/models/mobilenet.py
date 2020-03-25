import torch
from dlutils.models.nd import ConvNd, NormNd


class _ConvNormReLU(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, n_dim=2, norm_type="Batch"):
        padding = (kernel_size - 1) // 2

        super().__init__(
            ConvNd(n_dim, in_planes, out_planes, kernel_size, stride,
                   padding, groups=groups, bias=False),
            NormNd(norm_type, n_dim, out_planes),
            torch.nn.ReLU6(inplace=True)
        )


class _InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, n_dim=2,
                 norm_type="Batch"):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(_ConvNormReLU(inp, hidden_dim, kernel_size=1,
                                        n_dim=n_dim, norm_type=norm_type))
        layers.extend([
            # dw
            _ConvNormReLU(hidden_dim, hidden_dim, stride=stride,
                          groups=hidden_dim, n_dim=n_dim,
                          norm_type=norm_type),
            # pw-linear
            ConvNd(n_dim, hidden_dim, oup, 1, 1, 0, bias=False),
            NormNd(norm_type, n_dim, oup)
        ])
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 width_mult: float = 1.0, n_dim: int = 2,
                 norm_type: str = "Batch"):
        super().__init__()

        block = _InvertedResidual
        channels = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        channels = int(channels * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [_ConvNormReLU(in_channels, channels, stride=2, n_dim=n_dim,
                                  norm_type=norm_type)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(channels, output_channel, stride,
                                      expand_ratio=t, n_dim=n_dim,
                                      norm_type=norm_type))
                channels = output_channel
        # building last several layers
        features.append(_ConvNormReLU(channels, self.last_channel,
                                      kernel_size=1, n_dim=n_dim,
                                      norm_type=norm_type))
        # make it nn.Sequential
        self.features = torch.nn.Sequential(*features)

        # building classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.last_channel, num_classes),
        )

        self.squeeze_dims = list(range(2, n_dim + 2))
        self.reset_weights()

    def reset_weights(self):

        # weight initialization
        for m in self.modules():
            if isinstance(m, ConvNd):
                torch.nn.init.kaiming_normal_(m.conv.weight, mode='fan_out')
                if m.conv.bias is not None:
                    torch.nn.init.zeros_(m.conv.bias)
            elif isinstance(m, NormNd):
                if hasattr(m.norm, "weight") and m.norm.weight is not None:
                    torch.nn.init.ones_(m.norm.weight)

                if hasattr(m.norm, "bias") and m.norm.bias is not None:
                    torch.nn.init.zeros_(m.norm.bias)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(self.squeeze_dims)
        x = self.classifier(x)
        return x
