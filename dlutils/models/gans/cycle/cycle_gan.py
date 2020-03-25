import torch

from dlutils.models.gans.cycle.buffer import ReplayBuffer
from dlutils.models.gans.cycle.models import GeneratorResNet, \
    Discriminator


class CycleGAN(torch.nn.Module):
    """
    An implementation of cycle-consitent generative adversarial networks for
    unsupervised domain transfer with exchangeable generator and discriminator
    topologies

    References
    ----------
    `Paper <https://arxiv.org/abs/1606.07536>`_

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

    def __init__(self, input_shape, num_gen_blocks,
                 lambda_cycle, lambda_identity,
                 generator_cls=GeneratorResNet,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        input_shape : tuple
            the input shape (including channels, excluding batch dimension)
        num_gen_blocks : int
            the number of residual blocks to use for the generator
        lambda_cycle : float
            weighting factor for the cycle-consistency loss
        lambda_identity : float
            weighting factor for the identity loss
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """

        super().__init__()

        self.generator_a = generator_cls(input_channels=input_shape[0],
                                         num_residual_blocks=num_gen_blocks)

        self.generator_b = generator_cls(input_channels=input_shape[0],
                                         num_residual_blocks=num_gen_blocks)

        self.discriminator_a = discriminator_cls(input_shape)
        self.discriminator_b = discriminator_cls(input_shape)
        self.replay_buffer_a = ReplayBuffer()
        self.replay_buffer_b = ReplayBuffer()

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def forward(self, imgs_a: torch.Tensor, imgs_b: torch.Tensor):
        """
        Forwards the image batches through all relevant networks

        Parameters
        ----------
        imgs_a : :class:`torch.Tensor`
            the image batch of domain A
        imgs_b : :class:`torch.Tensor`
            the image batch of domain B

        Returns
        -------
        dict
            dictionary containing all (intermediate) outputs necessary for
            loss calculation and training

        """
        fake_b = self.generator_a(imgs_a)
        rec_a = self.generator_b(fake_b)
        id_b = self.generator_a(imgs_b)

        fake_a = self.generator_b(imgs_b)
        rec_b = self.generator_a(fake_a)
        id_a = self.generator_b(imgs_a)

        discr_real_a = self.discriminator_a(imgs_a)
        discr_real_b = self.discriminator_b(imgs_b)

        discr_fake_a = self.discriminator_a(fake_a)
        discr_fake_b = self.discriminator_b(fake_b)

        discr_fake_a_buffer = self.discriminator_a(
            self.replay_buffer_a(fake_a))
        discr_fake_b_buffer = self.discriminator_b(
            self.replay_buffer_b(fake_b))

        return {"fake_a": fake_a, "fake_b": fake_b, "rec_a": rec_a,
                "rec_b": rec_b, "discr_real_a": discr_real_a,
                "discr_real_b": discr_real_b, "discr_fake_a": discr_fake_a,
                "discr_fake_b": discr_fake_b, "id_a": id_a, "id_b": id_b,
                "discr_fake_a_buffer": discr_fake_a_buffer,
                "discr_fake_b_buffer": discr_fake_b_buffer}


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

    preds = model(data_dict["data_a", "data_b"])

    # Identity Loss
    loss_id_a = losses["identity"](preds["id_a"], data_dict["data_a"])
    loss_id_b = losses["identity"](preds["id_b"], data_dict["data_b"])

    loss_identity = (loss_id_a + loss_id_b) / 2

    # Adversarial GAN Loss
    loss_adv_a = losses["adversarial"](preds["discr_fake_a"], True)
    loss_adv_b = losses["adversarial"](preds["discr_fake_b"], True)

    loss_adversarial = (loss_adv_a + loss_adv_b) / 2

    # Cycle Consistency Loss
    loss_cycle_a = losses["cycle"](preds["rec_a"], data_dict["real_a"])
    loss_cycle_b = losses["cycle"](preds["rec_b"], data_dict["real_b"])

    loss_cycle = (loss_cycle_a + loss_cycle_b) / 2

    loss_gen = (loss_adversarial + attr_module.lambda_cycle * loss_cycle
                + attr_module.lambda_identity * loss_identity)

    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    loss_real_a = losses["adversarial"](preds["discr_real_a"], True)
    loss_fake_a = losses["adversarial"](preds["discr_fake_a_buffer"],
                                        False)

    loss_real_b = losses["adversarial"](preds["discr_real_b"], True)
    loss_fake_b = losses["adversarial"](preds["discr_fake_b_buffer"],
                                        False)

    loss_discr_a = (loss_real_a + loss_fake_a) / 2
    loss_discr_b = (loss_real_b + loss_fake_b) / 2

    optimizers["discriminator_a"].zero_grad()
    loss_discr_a.backward()
    optimizers["discriminator_a"].step()

    optimizers["discriminator_b"].zero_grad()
    loss_discr_b.backward()
    optimizers["discriminator_b"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
