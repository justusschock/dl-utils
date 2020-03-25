import torch


class AdversarialLoss(torch.nn.Module):
    def __init__(self, loss_fn=torch.nn.BCELoss(), same_size=False):
        # TODO: docstring
        super().__init__()
        self._loss_fn = loss_fn
        self._same_size = same_size

    def forward(self, pred: torch.Tensor, target: bool):
        if self._same_size:
            gt = torch.ones_like(pred)
        else:
            gt = torch.ones(pred.size(0), 1, device=pred.device,
                            dtype=pred.dtype)

        gt = gt * int(target)

        return self._loss_fn(pred, gt)
