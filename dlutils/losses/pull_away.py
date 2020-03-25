import torch


class PullAwayLoss(torch.nn.Module):
    """
    Pull Away Loss for the Energy-Based GANs
    References
    ----------
    `Paper <https://arxiv.org/abs/1609.03126>`_
    """

    def __init__(self, weight=1.):
        """
        Parameters
        ----------
        weight : float
            weight factor (specifying the impact compared to other loss
            functions)
        """
        super().__init__()
        self._weight = weight

    def forward(self, embeddings: torch.Tensor):
        """
        Parameters
        ----------
        embeddings : :class:`torch.Tensor`
            the embeddings of image batches
        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """
        norm = (embeddings ** 2).sum(-1, keepdim=True).sqrt()
        normalized_emb = embeddings / norm
        similarity = torch.matmul(normalized_emb,
                                  normalized_emb.transpose(0, 1))
        batchsize = embeddings.size(0)

        pt_loss = ((similarity.sum() - batchsize)
                   / (batchsize * (batchsize - 1)))

        return pt_loss * self._weight
