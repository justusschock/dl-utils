import torch

from dlutils.models.gans.wasserstein_div.models import Discriminator, \
    Generator


class WassersteinDivergenceGAN(torch.nn.Module):
    """
    Class implementing Wasserstein Divergence for GANs

    References
    ----------
    `Paper <https://arxiv.org/abs/1712.01026>`_

    Warnings
    --------
    This Network is designed for training only; if you want to predict from an
    already trained network, it might be best, to split this network into its
    parts (i. e. separating the discriminator from the generator). This will
    give a significant boost in inference speed and a significant decrease in
    memory consumption, since no memory is allocated for additional weights of
    the unused parts and no inference is done for them. If this whole network
    is used, inferences might be done multiple times per network, to obtain
    all necessary (intermediate) outputs for training.
    """

    def __init__(self, img_shape, latent_dim, gen_update_freq=5,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the real/generated images
        latent_dim : int
            size of the latent dimension
        gen_update_freq : int
            number of discriminator update steps to do per generator update
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """

        super().__init__()

        self.generator = generator_cls(img_shape, latent_dim)
        self.discriminator = discriminator_cls(img_shape)

        self._latent_dim = latent_dim
        self._update_gen_freq = gen_update_freq
        self._update_gen_ctr = 0

    def forward(self, x, z=None):
        """
        Feeds a set of batches through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        z : :class:`torch.Tensor`
            the noise batch, will be sampled from normal distribution if not
            given

        Returns
        -------
        dict
            a dictionary containing all relevant (intermediate) outputs
            necessary for loss calculation and training obtained by the
            different subnets

        """
        if z is None:
            z = torch.randn(x.size(0), self._latent_dim, device=x.device,
                            dtype=x.dtype)

        gen_imgs = self.generator(z)

        discr_fake = self.discriminator(gen_imgs)
        discr_real = self.discriminator(x)
        return {"gen_imgs": gen_imgs, "discr_fake": discr_fake,
                "discr_real": discr_real}

    @property
    def update_gen(self):
        """
        A property whether to update the generator in the current iteration

        Returns
        -------
        bool
            whether to update the generator

        """
        try:
            if self._update_gen_ctr == 0:
                return True
            return False
        # incrementing the counter will always be done because the finally
        # block is always executed a try-except block is exitted - even after
        # a return statement
        finally:
            self._update_gen_ctr = ((self._update_gen_ctr + 1)
                                    % self._update_gen_freq)


def update_fn(model, data_dict: dict, optimizers: dict, losses=None,
              ):
    """
    Function which handles prediction from batch, logging, loss calculation
    and optimizer step

    Parameters
    ----------
    model : torch.nn.Module
       model to forward data through
    data_dict : dict
       dictionary containing the data
    optimizers : dict
       dictionary containing all optimizers to perform parameter update
    losses : dict
       Functions or classes to calculate losses
    """

    if isinstance(model, torch.nn.DataParallel):
        attr_module = model.module
    else:
        attr_module = model

    # explicitly setting this to True is necessary to being able to
    # calculate gradients for Wasserstein Divergence
    data_dict["data"].requires_grad_(True)
    preds = model(data_dict["data"])

    update_gen = attr_module.update_gen

    loss_adv = preds["discr_fake"].mean() - preds["discr_real"].mean()
    loss_div = losses["divergence"](data_dict["data"], preds["discr_real"],
                                    preds["gen_imgs"], preds["discr_fake"])

    loss_d = loss_adv + loss_div

    optimizers["discriminator"].zero_grad()
    loss_d.backward(retain_graph=update_gen)
    optimizers["discriminator"].step()

    loss_gen = -preds["discr_fake"].mean()

    if update_gen:
        optimizers["generator"].zero_grad()
        loss_gen.backward()
        optimizers["generator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
