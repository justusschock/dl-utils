import torch


class GradientPenalty(torch.nn.Module):
    """
    A module to compute the gradient penalty
    """

    def forward(self, discr_interpolates: torch.Tensor,
                interpolates: torch.Tensor):
        """
        Computes the gradient penalty
        Parameters
        ----------
        discr_interpolates : :class:`torch.Tensor`
            the discriminator's output for the :param:`interpolates`
        interpolates : :class:`torch.Tensor`
            randomly distorted images as input for the discriminator
        Returns
        -------
        :class:`torch.Tensor`
            a weighted gradient norm
        """

        fake = torch.ones(interpolates.size(0), 1, device=interpolates.device,
                          dtype=interpolates.dtype)

        gradients = torch.autograd.grad(
            outputs=discr_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        return ((gradients.norm(p=2, dim=1) - 1) ** 2).mean()
