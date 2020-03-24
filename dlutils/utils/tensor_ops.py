import torch


def reduce(to_reduce: torch.Tensor, reduction: str):
    if reduction == 'elementwise_mean':
        return torch.mean(to_reduce)
    if reduction == 'none':
        return to_reduce
    if reduction == 'sum':
        return torch.sum(to_reduce)
    raise ValueError('Reduction parameter unknown.')


def make_onehot(labels, n_classes):
    """
    Function to convert a batch of class indices to onehot encoding
    Parameters
    ----------
    labels : torch.Tensor
        the batch of class indices
    n_classes : int
        the number of classes
    Returns
    -------
    torch.Tensor
        the onehot-encoded version of :param:`labels`
    """
    idx = labels.to(dtype=torch.long)

    new_shape = list(labels.unsqueeze(dim=1).shape)
    new_shape[1] = n_classes
    labels_onehot = torch.zeros(*new_shape, device=labels.device,
                                dtype=labels.dtype)
    labels_onehot.scatter_(1, idx.unsqueeze(dim=1), 1)
    return labels_onehot
