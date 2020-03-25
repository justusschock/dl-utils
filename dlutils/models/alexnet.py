import torch

from dlutils.models.nd import ConvNd, PoolingNd


class AlexNet(torch.nn.Module):
    def __init__(self, num_classes: int, in_channels: int, n_dim: int = 2,
                 pool_type: str = "Max"):
        super().__init__()
        self.features = torch.nn.Sequential(
            ConvNd(n_dim, in_channels, 64, kernel_size=11, stride=4,
                   padding=2),
            torch.nn.ReLU(inplace=True),
            PoolingNd(pool_type, n_dim, kernel_size=3, stride=2),
            ConvNd(n_dim, 64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            PoolingNd(pool_type, n_dim, kernel_size=3, stride=2),
            ConvNd(n_dim, 192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            ConvNd(n_dim, 384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            ConvNd(n_dim, 256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            PoolingNd(pool_type, n_dim, kernel_size=3, stride=2),
        )
        self.avgpool = PoolingNd("AdaptiveAvg", n_dim, 6)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * pow(6, n_dim), 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        features = self.features(input_batch)
        pooled = self.avgpool(features)
        flattened = pooled.view(input_batch.size(0), -1)
        return self.classifier(flattened)
