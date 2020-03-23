import torch
from typing import Sequence
from scipy.ndimage.morphology import distance_transform_edt
from dlutils.utils.tensor_ops import reduce

# TODO: Reimplement distance map transform in torch

__all__ = [
    'DistanceLoss',
    'distance_loss'
]


def get_dist_map(true_hot, in_object=None, factor=-1):
    distance_map = torch.zeros_like(true_hot)

    # calculates the euclidean distance map for each image and label
    # as the distance is calculated wrt. the background the mask has to be inverted
    for i in range(true_hot.shape[0]):
        for j in range(true_hot.shape[1]):
            distance_map[i, j] = torch.tensor(
                distance_transform_edt((true_hot[i, j] != 1).cpu().numpy()))
            # distance_map[i, j][true_hot[i, j] == 1] = in_object[j]
            distance_map[i, j][true_hot[i, j] == 1] = factor * distance_map[i, j].max()

    return distance_map


def distance_loss(predictions: torch.Tensor, targets: torch.Tensor,
                  weight: Sequence = None, reduction: str = 'elementwise_mean'):
    if weight is None or len(weight) != predictions.shape[1]:
        weight = torch.ones(predictions.shape[1], device=predictions.device,
                            dtype=predictions.dtype)

    weight_matrix = torch.stack([torch.ones_like(targets,
                                                 dtype=weight.dtype) * x
                                 for x in weight], 1)

    num_classes = predictions.shape[1]
    true_hot = torch.stack([targets == x
                            for x in range(num_classes)],
                           1).to(weight.dtype)

    # for the object factor=-1 to minimize the error on the object
    # by forcing the net to increase the respective class probability
    loss = predictions.softmax(dim=1) * get_dist_map(true_hot, factor=-1)
    loss = loss * weight_matrix / weight.sum()
    return reduce(loss, reduction=reduction)


class DistanceLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction="elementwise_mean", **kwargs):
        super().__init__()

        if weight is None:
            self.weight = weight
        else:
            self.register_buffer("weight", torch.tensor(weight))
        self.reduction = reduction

    def forward(self, predictions, targets):
        return distance_loss(predictions=predictions, targets=targets,
                             weight=self.weight, reduction=self.reduction)