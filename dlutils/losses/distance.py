import torch
from scipy.ndimage.morphology import distance_transform_edt

# TODO: Reimplement distance map transform in torch

__all__ = [
    'DistanceLoss'
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


class DistanceLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction="mean", **kwargs):
        super().__init__()

        if weight is None:
            self.weight = weight
        else:
            self.register_buffer("weight", torch.tensor(weight))
        self.reduction = reduction

    def forward(self, outputs, targets):
        # create a hot vector out of the targets for dimension 1
        # create a hot vector out of the targets for dimension 1
        weight = self.weight
        if weight is None or len(weight) != outputs.shape[1]:
            weight = torch.ones(outputs.shape[1], device=outputs.device,
                                dtype=outputs.dtype)

        weight_matrix = torch.stack([torch.ones_like(targets,
                                                     dtype=weight.dtype) * x
                                     for x in weight], 1)

        num_classes = outputs.shape[1]
        true_hot = torch.unsqueeze(targets, 1)
        true_hot = torch.stack([targets == x
                                for x in range(num_classes)],
                               1).to(weight.dtype)

        # for the object factor=-1 to minimize the error on the object
        # by forcing the net to increase the respective class probability
        loss = outputs.softmax(dim=1) * get_dist_map(true_hot, factor=-1)
        loss = loss * weight_matrix / weight.sum()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
