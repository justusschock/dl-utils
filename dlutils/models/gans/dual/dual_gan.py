import torch

from dlutils.models.gans.dual.models import Discriminator, GeneratorUNet


class DualGAN(torch.nn.Module):
    """
    Class implementing the Dual Generative Adversarial Networks and their
    behavior during training.

    References
    ----------
    `Paper <https://arxiv.org/abs/1704.02510>`_

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

    def __init__(self, img_shape: tuple, update_gen_iters=5,
                 lambda_cycle=10, lambda_adv=1,
                 generator_cls=GeneratorUNet,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of input batches (including channels, excluding batch
            dimension)
        update_gen_iters : int
            the generator will be updated once per ``update_gen_iters``
        lambda_cycle : float
            weighting factor for the cycle consistency loss
        lambda_adv : float
            weighting factor for the adversarial loss
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """

        super().__init__()

        self.generator_a = generator_cls(img_shape[0])
        self.generator_b = generator_cls(img_shape[0])

        self.discriminator_a = discriminator_cls(img_shape)
        self.discriminator_b = discriminator_cls(img_shape)
        self._update_gen_iters = update_gen_iters
        self._iter_ctr_gen = 0
        self.lambda_cycle = lambda_cycle
        self.lambda_adversarial = lambda_adv

    @property
    def update_generator(self) -> bool:
        """
        Property returning whether to update the generator during the current
        iteration

        Returns
        -------
        bool
            whether to update the gernator

        """
        self._iter_ctr_gen = (self._iter_ctr_gen + 1) % self._update_gen_iters

        return not bool(self._iter_ctr_gen)

    def forward(self, imgs_a, imgs_b, alpha_a=None, alpha_b=None):

        fake_a = self.generator_b(imgs_b)
        fake_b = self.generator_a(imgs_a)
        rec_a = self.generator_b(fake_b)
        rec_b = self.generator_a(fake_a)

        if alpha_a is None:
            # Random weight term for interpolation
            alpha_a = torch.randn_like(imgs_a)

        if alpha_b is None:
            alpha_b = torch.randn_like(imgs_b)

        interpolates_a = alpha_a * imgs_a + ((1 - alpha_a) * fake_a)
        interpolates_b = alpha_b * imgs_b + ((1 - alpha_b) * fake_b)

        discr_interpolates_a = self.discriminator_a(interpolates_a)
        discr_interpolates_b = self.discriminator_b(interpolates_b)

        discr_real_a = self.discriminator_a(imgs_a)
        discr_real_b = self.discriminator_b(imgs_b)

        discr_fake_a = self.discriminator_a(fake_a)
        discr_fake_b = self.discriminator_b(fake_b)

        return {"fake_a": fake_a, "fake_b": fake_b, "rec_a": rec_a,
                "rec_b": rec_b, "interpolates_a": interpolates_a,
                "interpolates_b": interpolates_b,
                "discr_interpolates_a": discr_interpolates_a,
                "discr_interpolates_b": discr_interpolates_b,
                "discr_real_a": discr_real_a, "discr_real_b": discr_real_b,
                "discr_fake_a": discr_fake_a, "discr_fake_b": discr_fake_b}


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
        attrmodule = model.module
    else:
        attrmodule = model

    update_gen = attrmodule.update_generator

    preds = model(data_dict["data_a"], data_dict["data_b"])

    # calculate losses for discriminator A
    gp_a = losses["gradient_penalty"](preds["discr_interpolates_a"],
                                      preds["interpolates_a"])

    loss_discr_real_a = -torch.mean(preds["discr_real_a"])
    loss_discr_fake_a = torch.mean(preds["discr_fake_a"])
    loss_discr_a = loss_discr_real_a + loss_discr_fake_a + gp_a

    # update discriminator A
    optimizers["discriminator_a"].zero_grad()
    loss_discr_a.backward(retain_graph=update_gen)
    optimizers["discriminator_a"].step()

    # calculate losses for discriminatr B
    gp_b = losses["gradient_penalty"](preds["discr_interpolates_b"],
                                      preds["interpolates_b"])
    loss_discr_real_b = -torch.mean(preds["discr_real_b"])
    loss_discr_fake_b = torch.mean(preds["discr_fake_b"])
    loss_discr_b = loss_discr_real_b + loss_discr_fake_b + gp_b

    # update discriminator B
    optimizers["discriminator_b"].zero_grad()
    loss_discr_b.backward(retain_graph=update_gen)
    optimizers["discriminator_b"].step()

    # calculate losses for generators
    loss_g_adv_a = -torch.mean(preds["discr_fake_a"])
    loss_g_adv_b = -torch.mean(preds["discr_fake_b"])
    loss_g_adv = loss_g_adv_a + loss_g_adv_b
    loss_g_cycle_a = losses["cycle"](preds["rec_a"], data_dict["data_a"])
    loss_g_cycle_b = losses["cycle"](preds["rec_b"], data_dict["data_b"])
    loss_g_cycle = loss_g_cycle_a + loss_g_cycle_b

    loss_g = (loss_g_adv * attrmodule.lambda_adversarial +
              loss_g_cycle * attrmodule.lambda_cycle)

    # update generators if adequate
    if update_gen:
        optimizers["generators"].zero_grad()
        loss_g.backward()
        optimizers["generators"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
