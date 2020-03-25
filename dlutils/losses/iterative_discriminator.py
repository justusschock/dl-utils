import torch


class IterativeDiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return sum([torch.mean((out - gt) ** 2) for out in preds])
