import torch


def reduce(to_reduce: torch.Tensor, reduction: str):
    if reduction == 'elementwise_mean':
        return torch.mean(to_reduce)
    if reduction == 'none':
        return to_reduce
    if reduction == 'sum':
        return torch.sum(to_reduce)
    raise ValueError('Reduction parameter unknown.')
