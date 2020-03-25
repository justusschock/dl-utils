import torch


class BoundarySeekingLoss(torch.nn.Module):
    """
    Boundary Seeking Loss
    References
    ----------
    https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """

    def __init__(self, weight=0.5):
        """
        Parameters
        ----------
        weight : float
            weighting factor
        """
        super().__init__()

        self._weight = weight

    def forward(self, pred):
        """
        Calculates the actual loss
        Parameters
        ----------
        pred : :class:`torch.Tensor`
            the prediction (typically obtained by the discriminator)
        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """
        return self._weight * torch.mean((torch.log(pred) -
                                          torch.log(1 - pred)) ** 2)
