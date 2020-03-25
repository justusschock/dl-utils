import torch


class WassersteinDivergence(torch.nn.Module):
    """
    Implements the Wasserstein Divergence proposed in
    `Wasserstein Divergence for GANS <https://arxiv.org/abs/1712.01026>`_
    """

    def __init__(self, p=6, k=2):
        """
        Parameters
        ----------
        p : int
            order of the norm
        k : int
            multiplicative factor applied to the mean
        """
        super().__init__()
        self._p = p
        self._k = k

    def forward(self, real_imgs, real_val, fake_imgs, fake_val):
        """
        Computes the actual divergence
        Parameters
        ----------
        real_imgs : :class:`torch.Tensor`
            the batch of real images
        real_val : :class:`torch.Tensor`
            the validity results for the real images obtained by feeding them
            through a discriminator
        fake_imgs : :class:`torch.Tensor`
            the batch of generated fake images
        fake_val : :class:`torch.Tensor`
            the validity results of the fake images obtained by feeding them
            through a discriminator
        Returns
        -------
        :class:`torch.Tensor`
            the wasserstein divergence
        """

        real_grad = torch.autograd.grad(
            real_val, real_imgs, torch.ones(real_imgs.size(0), 1,
                                            device=real_imgs.device,
                                            dtype=real_imgs.dtype,
                                            requires_grad=True),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        real_grad_norm = real_grad.norm(p=self._p)

        fake_grad = torch.autograd.grad(
            fake_val, fake_imgs, torch.ones(fake_imgs.size(0), 1,
                                            device=fake_imgs.device,
                                            dtype=fake_imgs.dtype,
                                            requires_grad=True),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]

        fake_grad_norm = fake_grad.norm(p=self._p)

        return torch.mean(real_grad_norm + fake_grad_norm) * self._k / 2
