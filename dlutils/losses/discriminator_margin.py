import torch


class DiscriminatorMarginLoss(torch.nn.Module):
    """
    A loss whose calculation switches slightly depending on a calculated
    margin.
    References
    --------
    `Paper <https://arxiv.org/abs/1609.03126>`_
    """

    def __init__(self, divisor=64., loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self._divisor = divisor
        self._loss_fn = loss_fn

    def forward(self, real_recon, real_imgs, fake_recon, fake_imgs):
        """
        Calculates the loss out of the given parameters
        Parameters
        ----------
        real_recon : :class:`torch.Tensor`
            the reconstruction of the real images
        real_imgs : :class:`torch.Tensor`
            the real image batch
        fake_recon : :class:`torch.Tensor`
            the reconstruction of the fake images
        fake_imgs : :class:`torch.Tensor`
            the (generated) fake image batch
        Returns
        -------
        :class:`torch.Tensor`
            the combined (margin-dependent) loss for real and fake images
        :class:`torch.Tensor`
            the loss only for real images
        :class:`torch.Tensor`
            the loss only for fake images
        """
        discr_real = self._loss_fn(real_recon, real_imgs)
        discr_fake = self._loss_fn(fake_recon, fake_imgs)

        margin = max(1., real_imgs.size(0) / self._divisor)

        discr_loss = discr_real

        if (margin - discr_fake).item() > 0:
            discr_loss += margin - discr_fake

        return discr_loss, discr_real, discr_fake
