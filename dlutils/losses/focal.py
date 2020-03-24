import torch

from torch.nn import functional as F
from rising.ops.tensor import torch_one_hot
from dlutils.utils.tensor_ops import reduce
from functools import reduce

__all__ = [
    'focal_loss',
    'focal_loss_with_logits',
    'binary_focal_loss',
    'binary_focal_loss_logits',
    'FocalLoss',
    'FocalLossWithLogits',
    'BinaryFocalLoss',
    'BinaryFocalLossWithLogits'
]


def _general_focal_loss(p: torch.Tensor, t: torch.Tensor, gamma: float,
                        loss_val: torch.Tensor, alpha_weight: float = 1.,
                        reduction: str = 'elementwise_mean'):
    """
    Helper Function Handling the general focal part and the reduction

    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    loss_val : torch.Tensor
        the value coming from the previous loss function
    alpha_weight : float
        class weight
    reduction : str
        reduction parameter

    Returns
    -------
    torch.Tensor
        loss value

    Raises
    ------
    ValueError
        invalid reduction parameter

    """
    # compute focal weights
    # if not isinstance(alpha_weight, torch.Tensor):
    #     alpha_weight = torch.tensor([1.], device=p.device)

    focal_weight = 1 - torch.where(torch.eq(t, 1.), p, 1 - p)
    focal_weight.pow_(gamma)
    focal_weight.to(p.device)

    # adjust shape if necessary
    if len(loss_val.shape) < len(focal_weight.shape):
        loss_val = loss_val.unsqueeze(1)

    # compute loss
    focal_loss = focal_weight * alpha_weight * loss_val

    return reduce(focal_loss, reduction)


def _bfocal_loss(p: torch.Tensor, t: torch.Tensor, loss_val: torch.Tensor,
                 gamma: float, alpha: float, reduction: str):
    """
    Helper function for binary focal loss

    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    loss_val : torch.Tensor
        the value coming from the previous loss function
    alpha : float
        class weight
    reduction : str
        reduction parameter

    Returns
    -------
    torch.Tensor
        loss value

    Raises
    ------
    ValueError
        invalid reduction parameter

    """
    if alpha is not None:
        # create weights for alpha
        alpha_weight = torch.ones(t.shape, device=p.device) * alpha
        alpha_weight = torch.where(torch.eq(t, 1.),
                                   alpha_weight, 1 - alpha_weight)
    else:
        alpha_weight = 1.

    # create weights for focal loss

    return _general_focal_loss(p=p, t=t, gamma=gamma, loss_val=loss_val,
                               alpha_weight=alpha_weight, reduction=reduction)


def binary_focal_loss(p: torch.Tensor, t: torch.Tensor, gamma: float = 2.,
                      alpha: float = None,
                      reduction: str = 'elementwise_mean'):
    """
    Binary focal loss without (!) logits
    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    alpha : float
        class weight
    reduction : str
        reduction parameter

    Returns
    -------
    torch.Tensor
        loss value

    Raises
    ------
    ValueError
        invalid reduction parameter

    """
    loss_val = F.binary_cross_entropy(p, t, reduction='none')
    return _bfocal_loss(p=p, t=t, loss_val=loss_val, gamma=gamma, alpha=alpha,
                        reduction=reduction)


def binary_focal_loss_logits(p: torch.Tensor, t: torch.Tensor,
                             gamma: float = 2., alpha: float = None,
                             reduction: str = 'elementwise_mean'):
    """
    Binary focal loss with logits
    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    alpha : float
        class weight
    reduction : str
        reduction parameter

    Returns
    -------
    torch.Tensor
        loss value

    Raises
    ------
    ValueError
        invalid reduction parameter

    """
    loss_val = F.binary_cross_entropy_with_logits(
        p, t, reduction='none')

    p = torch.sigmoid(p)
    return _bfocal_loss(p=p, t=t, loss_val=loss_val, gamma=gamma, alpha=alpha,
                        reduction=reduction)


def _focal_loss(p: torch.Tensor, t: torch.Tensor, gamma: float,
                loss_val: torch.Tensor, reduction: str):
    """
    Focal loss helper function
    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    loss_val : torch.Tensor
        value coming from the previous (weighted) loss function
    reduction : str
        reduction parameter

    Returns
    -------
    torch.Tensor
        loss value

    Raises
    ------
    ValueError
        invalid reduction parameter

    """
    n_classes = p.size(1)
    target_onehot = torch_one_hot(t, num_classes=n_classes)
    return _general_focal_loss(p=p, t=target_onehot, gamma=gamma,
                               loss_val=loss_val, reduction=reduction,
                               alpha_weight=1.)


def focal_loss(p: torch.Tensor, t: torch.Tensor, gamma: float = 2.,
               alpha: torch.Tensor = None,
               reduction: str = 'elementwise_mean'):
    """
    Focal loss without (!) logits

    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    alpha : torch.Tensor
        class weight
    reduction : str
        reduction parameter

    Returns
    -------
    torch.Tensor
        loss value

    Raises
    ------
    ValueError
        invalid reduction parameter

    """
    loss_val = F.nll_loss(p, t, weight=alpha, reduction='none')
    p = p.log()
    return _focal_loss(p=p, t=t, gamma=gamma, reduction=reduction,
                       loss_val=loss_val)


def focal_loss_with_logits(p: torch.Tensor, t: torch.Tensor, gamma: float = 2.,
                           alpha: torch.Tensor = None,
                           reduction: str = 'elementwise_mean'):
    """
    focal loss with logits
    Parameters
    ----------
    p: torch.Tensor
        the prediction tensor
    t : torch.Tensor
        the target tensor
    gamma : float
        focusing parameter
    alpha : torch.Tensor
        class weight
    reduction : str
        reduction parameter

    Returns
    -------
    torch.Tensor
        loss value

    Raises
    ------
    ValueError
        invalid reduction parameter

    """
    loss_val = F.cross_entropy(p, t, weight=alpha, reduction='none')
    p = F.softmax(p, dim=1)
    return _focal_loss(p=p, t=t, gamma=gamma, reduction=reduction,
                       loss_val=loss_val)


class BinaryFocalLoss(torch.nn.Module):
    """
    Focal loss for binary case without(!) logit
    """

    def __init__(self, alpha=None, gamma=2, reduction='elementwise_mean'):
        """
        Implements Focal Loss for binary classification case
        Parameters
        ----------
        alpha : float
            alpha has to be in range [0,1], assigns class weight
        gamma : float
            focusing parameter
        reduction : str
            Specifies the reduction to apply to the output: ‘none’ |
            ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
            ‘elementwise_mean’: the sum of the output will be divided by the
            number of elements in the output, ‘sum’: the output will be summed
        (further information about parameters above can be found in pytorch
        documentation)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, p, t):
        """
        Compute Focal Loss with logits
        Parameters
        ----------
        p : torch.Tensor
            prediction
        t : torch.Tensor
            target
        Returns
        -------
        torch.Tensor
            result
        """
        return binary_focal_loss(p=p, t=t, gamma=self.gamma, alpha=self.alpha,
                                 reduction=self.reduction)


class BinaryFocalLossWithLogits(torch.nn.Module):
    """
    Focal loss for binary case WITH logits
    """

    def __init__(self, alpha=None, gamma=2, reduction='elementwise_mean'):
        """
        Implements Focal Loss for binary class case
        Parameters
        ----------
        alpha : float
            alpha has to be in range [0,1], assigns class weight
        gamma : float
            focusing parameter
        reduction : str
            Specifies the reduction to apply to the output: ‘none’ |
            ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
            ‘elementwise_mean’: the sum of the output will be divided by the
            number of elements in the output, ‘sum’: the output will be summed
        (further information about parameters above can be found in pytorch
        documentation)
        Returns
        -------
        torch.Tensor
            loss value
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, p, t):
        """
        Compute Focal Loss with logits
        Parameters
        ----------
        p : torch.Tensor
            prediction
        t : torch.Tensor
            target
        Returns
        -------
        torch.Tensor
            result
        """

        return binary_focal_loss_logits(p=p, t=t, alpha=self.alpha,
                                        gamma=self.gamma,
                                        reduction=self.reduction)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):
        """
        Implements Focal Loss for non-binary class case
        Parameters
        ----------
        alpha : torch.Tensor
            alpha has to be in range [0,1], assigns class weight
        gamma : float
            focusing parameter
        reduction : str
            Specifies the reduction to apply to the output: ‘none’ |
            ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
            ‘elementwise_mean’: the sum of the output will be divided by the
            number of elements in the output, ‘sum’: the output will be summed
        (further information about parameters above can be found in pytorch
        documentation)
        Returns
        -------
        torch.Tensor
            loss value
        """
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = alpha
        self.reduction = reduction

    def forward(self, p, t):
        """
        Compute Focal Loss without logits
        Parameters
        ----------
        p : torch.Tensor
            prediction
        t : torch.Tensor
            target
        Returns
        -------
        torch.Tensor
            result
        """
        return focal_loss(p=p, t=t, gamma=self.gamma, alpha=self.alpha,
                          reduction=self.reduction)


class FocalLossWithLogits(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):
        """
        Implements Focal Loss with logits for non-binary class case
        Parameters
        ----------
        alpha : torch.Tensor
            alpha has to be in range [0,1], assigns class weight
        gamma : float
            focusing parameter
        reduction : str
            Specifies the reduction to apply to the output: ‘none’ |
            ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
            ‘elementwise_mean’: the sum of the output will be divided by the
            number of elements in the output, ‘sum’: the output will be summed
        (further information about parameters above can be found in pytorch
        documentation)
        Returns
        -------
        torch.Tensor
            loss value
        """
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = alpha
        self.reduction = reduction

    def forward(self, p, t):
        """
        Compute Focal Loss with logits
        Parameters
        ----------
        p : torch.Tensor
            prediction
        t : torch.Tensor
            target
        Returns
        -------
        torch.Tensor
            result
        """
        return focal_loss_with_logits(p=p, t=t, gamma=self.gamma,
                                      alpha=self.alpha,
                                      reduction=self.reduction)
