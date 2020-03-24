from typing import Union

import torch


def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    reduces a given tensor by a given reduction method

    Parameters
    ----------
    to_reduce : torch.Tensor
        the tensor, which shall be reduced
    reduction : str
        a string specifying the reduction method.
        should be one of 'elementwise_mean' | 'none' | 'sum'

    Returns
    -------
    torch.Tensor
        reduced Tensor

    Raises
    ------
    ValueError
        if an invalid reduction parameter was given

    """
    if reduction == 'elementwise_mean':
        return torch.mean(to_reduce)
    if reduction == 'none':
        return to_reduce
    if reduction == 'sum':
        return torch.sum(to_reduce)
    raise ValueError('Reduction parameter unknown.')


def atleast_1d(*tensors) -> Union[torch.Tensor, list]:
    """
    Convert inputs to tensors with at least one dimension.

    Scalar inputs are converted to 1-dimensional tensors, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    tensor1, tensor2, ... : tensor_like
        One or more input tensors.

    Returns
    -------
    torch.Tensor or list
        A tensor, or list of tensors, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    """
    res = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        if tensor.ndim == 0:
            result = tensor.view(1)
        else:
            result = tensor
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def make_onehot(labels, num_classes):
    """
    Function to convert a batch of class indices to onehot encoding
    Parameters
    ----------
    labels : torch.Tensor
        the batch of class indices
    num_classes : int
        the number of classes
    Returns
    -------
    torch.Tensor
        the onehot-encoded version of :param:`labels`
    """
    idx = labels.to(dtype=torch.long)

    new_shape = list(labels.unsqueeze(dim=1).shape)
    new_shape[1] = num_classes
    labels_onehot = torch.zeros(*new_shape, device=labels.device,
                                dtype=labels.dtype)
    labels_onehot.scatter_(1, idx.unsqueeze(dim=1), 1)
    return labels_onehot
