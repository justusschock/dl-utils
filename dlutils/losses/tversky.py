from typing import Callable
import torch
from rising.transforms.functional.channel import one_hot_batch
from dlutils.utils.tensor_ops import reduce
from functools import reduce


def tversky_loss(predictions: torch.Tensor, targets: torch.Tensor,
                 alpha: float = 0.5, beta: float = 0.5,
                 weight: torch.Tensor = None,
                 non_lin: Callable = None, square_nom: bool = False,
                 square_denom: bool = False, smooth: float = 1.,
                 reduction: str = 'elementwise_mean') -> torch.Tensor:
    """
    Calculates the tversky loss

    Parameters
    ----------
    predictions : torch.Tensor
        the predicted segmentation (of shape NxCx(Dx)HxW)
    targets : torch.Tensor
        the groundtruth segmentation (of shape Nx(Dx)HxW
    alpha : float
        scaling factor for false negatives
    beta : float
        scaling factor for false positives
    weight : torch.Tensor
        weighting factors for each class
    non_lin : Callable
        a non linearity to apply on the predictions before calculating
        the loss value
    square_nom : bool
        whether to square the nominator
    square_denom : bool
        whether to square the denominator
    smooth : float
        smoothing value (to avid divisions by 0)
    reduction : str
        kind of reduction to apply to the final loss

    Returns
    -------
    torch.Tensor
        reduced loss value

    """
    n_classes = predictions.shape[1]
    dims = tuple(range(2, predictions.dim()))

    if non_lin is not None:
        predictions = non_lin(predictions)

    target_onehot = one_hot_batch(targets.unsqueeze(1), num_classes=n_classes)
    target_onehot = target_onehot.float()

    tp = predictions * target_onehot
    fp = predictions * (1 - target_onehot)
    fn = (1 - predictions) * target_onehot

    if square_nom:
        tp = tp ** 2
    if square_denom:
        fp = fp ** 2
        fn = fn ** 2

    # compute nominator
    tp_sum = torch.sum(tp, dim=dims)
    nom = tp_sum + smooth

    # compute denominator
    denom = tp_sum + alpha * torch.sum(fn, dim=dims) + \
        beta * torch.sum(fp, dim=dims) + smooth

    # compute loss
    frac = nom / denom

    # apply weights to individual classes
    if weight is not None:
        frac = weight * frac

    # average over classes
    frac = 1 - torch.mean(frac, dim=1)
    return reduce(-frac, reduction)


def focal_tversky(predictions: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.33,
                  weight: torch.Tensor = None,
                  non_lin: Callable = None, square_nom: bool = False,
                  square_denom: bool = False, smooth: float = 1.,
                  reduction: str = 'elementwise_mean') -> torch.Tensor:
    """
    Calculates the focal tversky loss

    Parameters
    ----------
    predictions : torch.Tensor
        the predicted segmentation (of shape NxCx(Dx)HxW)
    targets : torch.Tensor
        the groundtruth segmentation (of shape Nx(Dx)HxW
    alpha : float
        scaling factor for false negatives
    beta : float
        scaling factor for false positives
    gamma: float
        focusing factor
    weight : torch.Tensor
        weighting factors for each class
    non_lin : Callable
        a non linearity to apply on the predictions before calculating
        the loss value
    square_nom : bool
        whether to square the nominator
    square_denom : bool
        whether to square the denominator
    smooth : float
        smoothing value (to avid divisions by 0)
    reduction : str
        kind of reduction to apply to the final loss

    Returns
    -------
    torch.Tensor
        reduced loss value

    """
    loss = tversky_loss(predictions=predictions, targets=targets, alpha=alpha,
                        beta=beta, weight=weight, non_lin=non_lin,
                        square_denom=square_denom, square_nom=square_nom,
                        smooth=smooth, reduction='none')

    focal = loss ** gamma
    return reduce(focal, reduction)
