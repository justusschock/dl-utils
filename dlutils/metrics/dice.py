import torch
import logging

logger = logging.getLogger(__file__)

__all__ = [
    'compute_stat_score',
    'dice_score',
    'binary_dice_coefficient',
    'DiceCoefficient',
    'BinaryDiceCoefficient'
]


def binary_dice_coefficient(pred: torch.Tensor, gt: torch.Tensor,
                            thresh: float = 0.5, smooth: float = 1e-7):
    """
    A binary dice coefficient

    Parameters
    ----------
    pred : torch.Tensor
        predicted segmentation (of shape NxCx(Dx)HxW)
    gt : torch.Tensor
        target segmentation (of shape NxCx(Dx)HxW)
    thresh : float
        segmentation threshold
    smooth : float
        smoothing value to avoid division by zero

    Returns
    -------
    torch.Tensor
        dice score

    """
    pred_bool = pred > thresh

    intersec = (pred_bool and gt).float()
    return 2 * intersec.sum() / (pred_bool.float().sum()
                                 + gt.float().sum() + smooth)


def compute_stat_score(pred, label, n, do_argmax: bool = True):
    """Compute metric statistics (true positive, false positive, true negative,
    false negative)
    Parameters
    ----------
    pred : torch.Tensor
        network predictions with probability for each class
    label : torch.Tensor
        ground truth annotation. Classes are given by numeric value
    n : int
        compute scores for class n
    Returns
    -------
    int
        number of true positives
    int
        number of false positives
    int
        number of true negatives
    int
        number of false negatives
    """
    if do_argmax:
        cls_pred = torch.argmax(pred, dim=1)
    else:
        cls_pred = pred[:, 0] > 0.5

    tp = ((cls_pred == n) * (label == n)).to(torch.long).sum()
    fp = ((cls_pred == n) * (label != n)).to(torch.long).sum()
    tn = ((cls_pred != n) * (label != n)).to(torch.long).sum()
    fn = ((cls_pred != n) * (label == n)).to(torch.long).sum()

    return tp, fp, tn, fn


def dice_score(pred: torch.Tensor, label: torch.Tensor, bg=False, cls_logging=False,
               nan_score=0.0, no_fg_score=0.0, apply_argmax: bool = True):
    """
    Compute dice score 1/n_classes * (2*tp)/(2*tp + fp + fn)
    Parameters
    ----------
    pred : torch.Tensor
        probability for each class
    label : torch.Tensor
        ground truth annotation. Classes are given by numeric value
        (not onehot encoded)
    bg : bool, optional
        compute dice for background class, by default False
    cls_logging : int, optional
        logging for individual class results
    nan_score: float, optional
        if denominator is zero `nan_score`is used instead.
    no_fg_score: float, optional
        if foreground class is not present, `np_fg_score` is sued instead.
    Returns
    -------
    float
        dice score
    """
    if not (pred > 0).any():
        logger.warning(
            "Prediction only contains zeros. Dice score might be ambigious.")

    # invert background value
    bg = (1 - int(bool(bg)))

    n_classes = pred.shape[1]
    score = 0.
    for i in range(bg, n_classes):
        tp, fp, tn, fn = compute_stat_score(pred, label, i, do_argmax=apply_argmax)

        denom = (2 * tp + fp + fn).to(torch.float)
        if not (label == i).any():
            # no foreground class
            score_cls = no_fg_score
        elif torch.isclose(denom, torch.zeros_like(denom)).any():
            # nan result
            score_cls = nan_score
        else:
            score_cls = (2 * tp).to(torch.float) / denom

        if cls_logging:
            logger.info({'value': {'value': score_cls,
                                   'name': 'dice_cls_' + str(i)}})
            pass

        score += score_cls
    return score / (n_classes - bg)


class DiceCoefficient(torch.nn.Module):
    def __init__(self, bg=False, cls_logging=False,
                 nan_score=0.0, no_fg_score=0.0, apply_argmax: bool = True,
                 ):
        """

        Compute dice score 1/n_classes * (2*tp)/(2*tp + fp + fn)
        Parameters
        ----------
        pred : torch.Tensor
            probability for each class
        label : torch.Tensor
            ground truth annotation. Classes are given by numeric value
            (not onehot encoded)
        bg : bool, optional
            compute dice for background class, by default False
        cls_logging : int, optional
            logging for individual class results
        nan_score: float, optional
            if denominator is zero `nan_score`is used instead.
        no_fg_score: float, optional
            if foreground class is not present, `np_fg_score` is sued instead.

        """
        super().__init__()
        self.bg = bg
        self.cls_logging = cls_logging
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        self.apply_argmax = apply_argmax

    def forward(self, predictions, targets):
        return dice_score(predictions, targets, bg=self.bg,
                          cls_logging=self.cls_logging,
                          nan_score=self.nan_score,
                          no_fg_score=self.no_fg_score,
                          apply_argmax=self.apply_argmax)


class BinaryDiceCoefficient(torch.nn.Module):
    def __init__(self, thresh: float = 0.5, smooth: float = 1e-7):
        """
        Compute dice score for binary classification tasks

        Parameters
        ----------
        thresh : float
            segmentation threshold
        smooth : float
            smoothing value to avoid division by zero
        """
        super().__init__()
        self.thresh = thresh
        self.smooth = smooth

    def forward(self, predictions, targets):
        return binary_dice_coefficient(pred=predictions, gt=targets,
                                       thresh=self.thresh, smooth=self.smooth)
