import torch

from dlutils.models.gans.star.models import Discriminator, GeneratorResNet


class StarGAN(torch.nn.Module):
    """
    Class implementing StarGAN: Unified Generative Adversarial Networks for
    Multi-Domain Image-to-Image Translation

    References
    ----------
    `Paper <https://arxiv.org/abs/1711.09020>`_

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

    def __init__(self, img_shape, c_dim=5, num_res_blocks=9, num_strided=6,
                 gen_update_freq=5, lambda_cls=1., lambda_rec=10.,
                 lambda_gp=10., generator_cls=GeneratorResNet,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the generated/input images; should contain channels,
            but not the batch dimension
        c_dim : int
            size of the latent code dimension
        num_res_blocks : int
            number of residual blocks inside the generator
        num_strided : int
            number of strided convolutions inside the discriminator
        gen_update_freq : int
            number of discriminator updates before updating the generator once
        lambda_cls : float
            the impact factor of the classification loss
        lambda_rec : float
            the impact factor of the recovery (cycle consistency) loss
        lambda_gp : float
            the impact factor of the gradient penalty loss
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """
        super().__init__()

        self.generator = generator_cls(img_shape, num_res_blocks, c_dim)
        self.discriminator = discriminator_cls(img_shape, c_dim, num_strided)

        self._c_dim = c_dim
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_gp = lambda_gp
        self._gen_update_freq = gen_update_freq
        self._gen_update_ctr = 0

    @property
    def update_generator(self):
        """
        Property to indicate whether to update the generator in the current
        iteration

        Returns
        -------
        bool
            True, if generator should be updated; Else False
        """

        try:
            if self._gen_update_ctr == 0:
                return True
            return False
        finally:
            self._gen_update_ctr = ((self._gen_update_ctr + 1)
                                    % self._gen_update_freq)

    def forward(self, x, labels, c=None, alpha=None):
        """
        Forwards a set of batches through the relevant networks

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        labels : :class:`torch.Tensor`
            the batch containing the labels
        c : :class:`torch.Tensor`
            the batch containing the encoded target attributes, will be
            randomly sampled if not given
        alpha : :class:`torch.Tensor`
            the alpha value for interpolation between real and fake images;
            Will be sampled if not given

        Returns
        -------
        dict
            dictionary containing all relevant (intermediate) outputs
            necessary for loss calculation and training

        """

        if c is None:
            c = torch.randint(0, 2, (x.size(0), self._c_dim), device=x.device)

        fake_imgs = self.generator(x, c)
        recov_imgs = self.generator(fake_imgs, labels)

        fake_val, fake_cls = self.discriminator(fake_imgs)
        real_val, real_cls = self.discriminator(x)

        if alpha is None:
            alpha = torch.rand(x.size(0), 1, 1, 1)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * x + ((1 - alpha) * fake_imgs))
        interpolates.requires_grad_(True)

        discr_interpolates, _ = self.discriminator(interpolates)

        return {"fake_imgs": fake_imgs, "rec_imgs": recov_imgs,
                "fake_val": fake_val, "fake_cls": fake_cls,
                "real_val": real_val, "real_cls": real_cls,
                "discr_interpolates": discr_interpolates,
                "interpolates": interpolates, "sampled_c": c}


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

    preds = model(data_dict["data"], data_dict["label"])

    discr_gp = losses["gradient_penalty"](preds["discr_interpolates"],
                                          preds["interpolates"])

    discr_adv = -preds["real_val"].mean() + preds["fake_val"].mean()

    discr_cls = losses["auxiliary"](preds["real_cls"], data_dict["label"])

    loss_discr = (discr_adv
                  + attr_module.lambda_gp * discr_gp
                  + attr_module.lambda_cls * discr_cls)

    # check whether to update generator and save result to variable to
    # avoid increasing the counter multiple times
    update_gen = attr_module.update_generator
    optimizers["discriminator"].zero_grad()
    loss_discr.backward(retain_graph=update_gen)
    optimizers["discriminator"].step()

    gen_adv = -preds["fake_val"].mean()

    gen_cls = losses["auxiliary"](preds["fake_cls"], preds["sampled_c"])

    gen_rec = losses["cycle"](preds["rec_imgs"], data_dict["data"])

    loss_gen = (gen_adv
                + attr_module.lambda_cls * gen_cls
                + attr_module.lambda_rec * gen_rec)

    if attr_module.update_generator:
        optimizers["generator"].zero_grad()
        loss_gen.backward()
        optimizers["generator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
