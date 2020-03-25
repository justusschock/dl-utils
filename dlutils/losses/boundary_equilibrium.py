import torch


class BELoss(torch.nn.Module):
    """
    Boundary Equilibrium Loss
    """

    def __init__(self, gamma=0.75, lambda_k=0.001, initial_k=0.0):
        """
        Parameters
        ----------
        gamma : float
            impact of real_loss on weight update
        lambda_k : float
            impact of loss difference on weight update
        initial_k : float
            initial weight value
        """
        super().__init__()
        self._k = initial_k
        self._gamma = gamma
        self._lambda_k = lambda_k

    def forward(self, discr_real, real_imgs, discr_fake, gen_imgs):
        """
        Computes the losses
        Parameters
        ----------
        discr_real : :class:`torch.Tensor`
            the discriminiators output for real images
        real_imgs : :class:`torch.Tensor`
            the real images
        discr_fake : :class:`torch.Tensor`
            the discriminator output for generated images
        gen_imgs : :class:`torch.Tensor`
            the generated images
        Returns
        -------
        :class:`torch.Tensor`
            the total loss
        :class:`torch.Tensor`
            the part of the total loss coming from real images
            (without weighting)
        :class:`torch.Tensor`
            the part of the loss coming from fake images (without weighting)
        """
        loss_real = self._loss_fn(discr_real, real_imgs)
        loss_fake = self._loss_fn(discr_fake, gen_imgs.detach())

        total_loss = loss_real - self._k * loss_fake

        # update weight term
        diff = (self._gamma * loss_real - loss_fake).mean().item()
        self._k = self._k + self._lambda_k * diff
        # constrain to [0, 1]
        self._k = min(max(self._k, 0), 1)

        return total_loss, loss_real, loss_fake

    @staticmethod
    def _loss_fn(pred, label):
        """
        The actual loss function; Computes Mean L1-Error
        Parameters
        ----------
        pred : :class:`torch.Tensor`
            the predictions
        label : :class:`torch.Tensor`
            the labels
        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """
        return (pred - label).abs().mean()
