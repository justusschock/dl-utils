import torch

from dlutils.models.gans.unit.models import Generator, Encoder, \
    Discriminator, ResidualBlock


class UNIT(torch.nn.Module):
    """
    Class implementing Unsupervised Image-to-Image Translation Networks

    References
    ----------
    `Paper <https://arxiv.org/abs/1703.00848>`_

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

    def __init__(self, img_shape, num_filts, n_sample,
                 shared_dim=None, lambda_adv=10., lambda_kl=0.1,
                 lambda_id=100., lambda_fake_kl=0.1, lambda_cycle=100.,
                 block_cls=ResidualBlock, generator_cls=Generator,
                 encoder_cls=Encoder, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the images to translate from and to (including
            channels, excluding batch dimension)
        num_filts : int
            number of filters to use
        n_sample : int
            number of sampling layers per network
        shared_dim : int
            size of the shared dimension between generators and between
            encoders; If not given, this will be calculated based on
            :param:`n_sample` and :param:`num_filts`
        lambda_adv : float
            impact factor of the adversarial loss
        lambda_kl : float
            impact factor of the KL loss
        lambda_id : float
            impact factor of the identity loss
        lambda_fake_kl
         : float
            impact factor of the reconstructed KL loss
        lambda_cycle : float
            impact factor of the cycle consistency loss
        block_cls :
            class implementing the actual block topology
        generator_cls :
            class implementing the actual generator topology
        encoder_cls :
            class implementing the actual encoder topology
        discriminator_cls :
            class implementing the actual deocder topology
        """

        super().__init__()

        if shared_dim is None:
            shared_dim = num_filts * 2 ** n_sample
        shared_block_enc = block_cls(shared_dim)
        shared_block_gen = block_cls(shared_dim)
        self.encoder_a = encoder_cls(shared_block_enc, img_shape[0], num_filts,
                                     n_sample)
        self.encoder_b = encoder_cls(shared_block_enc, img_shape[0], num_filts,
                                     n_sample)

        self.generator_a = generator_cls(shared_block_gen, img_shape[0],
                                         num_filts, n_sample)
        self.generator_b = generator_cls(shared_block_gen, img_shape[0],
                                         num_filts, n_sample)

        self.discriminator_a = discriminator_cls(img_shape)
        self.discriminator_b = discriminator_cls(img_shape)

        self.lambda_adversarial = lambda_adv
        self.lambda_kl = lambda_kl
        self.lambda_identity = lambda_id
        self.lambda_fake_kl = lambda_fake_kl
        self.lambda_cycle = lambda_cycle

    def forward(self, imgs_a, imgs_b):
        """
        Feeds a set of image batches through all necessary networks

        Parameters
        ----------
        imgs_a : :class:`torch.Tensor`
            the images of domain A
        imgs_b : :class:`torch.Tensor`
            the images of domain B

        Returns
        -------
        dict
            a dictionary containing all the relevant (intermediate) outputs
            necessary for loss calculation and training

        """
        mu_a, z_a = self.encoder_a(imgs_a)
        mu_b, z_b = self.encoder_b(imgs_b)

        rec_a = self.generator_a(z_a)
        rec_b = self.generator_b(z_b)

        fake_a = self.generator_a(z_b)
        fake_b = self.generator_b(z_a)

        fake_mu_a, fake_z_a = self.encoder_a(fake_a)
        fake_mu_b, fake_z_b = self.encoder_b(fake_b)

        cyc_a = self.generator_a(fake_z_b)
        cyc_b = self.generator_b(fake_z_a)

        discr_fake_a = self.discriminator_a(fake_a)
        discr_fake_b = self.discriminator_b(fake_b)

        discr_real_a = self.discriminator_a(imgs_a)
        discr_real_b = self.discriminator_b(imgs_b)

        return {"mu_a": mu_a, "mu_b": mu_b, "z_a": z_a, "z_b": z_b,
                "rec_a": rec_a, "rec_b": rec_b, "fake_a": fake_a,
                "fake_b": fake_b, "fake_mu_a": fake_mu_a, "fake_z_a": fake_z_a,
                "fake_mu_b": fake_mu_b, "fake_z_b": fake_z_b, "cyc_a": cyc_a,
                "cyc_b": cyc_b, "discr_fake_a": discr_fake_a,
                "discr_fake_b": discr_fake_b, "discr_real_a": discr_real_a,
                "discr_real_b": discr_real_b}


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

    preds = model(data_dict["imgs_a"], data_dict["imgs_b"])

    # calculate adversarial loss
    adv_a = losses["adversarial"](preds["discr_fake_a"], True)
    adv_b = losses["adversarial"](preds["discr_fake_b"], True)

    # calculate kl loss
    kl_a = (preds["mu_a"] ** 2).mean()
    kl_b = (preds["mu_b"] ** 2).mean()

    # calculate identity loss
    id_a = losses["pixelwise"](preds["rec_a"], data_dict["imgs_a"])
    id_b = losses["pixelwise"](preds["rec_b"], data_dict["imgs_b"])

    # calculate kl loss for fakes
    kl_fake_a = (preds["mu_fake_a"] ** 2).mean()
    kl_fake_b = (preds["mu_fake_b"] ** 2).mean()

    # calculate cycle consistency loss
    cyc_a = losses["pixelwise"](preds["cyc_a"], data_dict["imgs_a"])
    cyc_b = losses["pixelwise"](preds["cyc_b"], data_dict["imgs_b"])

    # calculate total generator loss
    loss_gen = (
            attr_module.lambda_adversarial * (adv_a + adv_b)
            + attr_module.lambda_kl * (kl_a + kl_b)
            + attr_module.lambda_identity * (id_a + id_b)
            + attr_module.lambda_fake_kl * (kl_fake_a + kl_fake_b)
            + attr_module.lambda_cycle * (cyc_a + cyc_b)
    )

    # update generator
    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    # calculate adversarial losses and total discriminator losses
    discr_a_real = losses["adversarial"](preds["discr_real_a"], True)
    discr_b_real = losses["adversarial"](preds["discr_real_b"], True)
    discr_a_fake = losses["adversarial"](preds["discr_fake_a"], False)
    discr_b_fake = losses["adversarial"](preds["discr_fake_b"], False)
    discr_a = discr_a_real + discr_a_fake
    discr_b = discr_b_real + discr_b_fake

    # optimize discriminators
    optimizers["discriminator_a"].zero_grad()
    discr_a.backward(retain_graph=True)
    optimizers["discriminator_a"].step()

    optimizers["discriminator_b"].zero_grad()
    discr_b.backward(retain_graph=True)
    optimizers["discriminator_b"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
