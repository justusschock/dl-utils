import torch

from dlutils.models.gans.adversarial_autoencoder.models import \
    Generator, Discriminator


class AdversarialAutoEncoderPyTorch(torch.nn.Module):
    """
    Class implementing the Combined Adversarial Autoencoder and it's behavior
    during training.

    An adversarial autoencoder is basically aprobabilistic autoencoder that
    uses generative adversarial networks (GAN) to perform variational inference
    by matching the aggregated posterior of the hidden code vector of the
    autoencoder with an arbitrary prior distribution

    References
    ----------
    `Paper <https://arxiv.org/abs/1511.05644>`_

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

    def __init__(self, latent_dim, img_shape, generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        latent_dim : int
            the size of the autoencoders latend dimension
        img_shape : tuple
            the shape of the input/output image
        generator_cls :
            a class implementing the actual generator model (consisting of
            encoder and decoder)
        discriminator_cls :
            a class implementing the actual discriminator model

        """
        super().__init__()

        self.generator = generator_cls(latent_dim=latent_dim,
                                       img_shape=img_shape)

        self.discriminator = discriminator_cls(latent_dim=latent_dim)
        self._latent_dim = latent_dim

    def forward(self, x):
        """
        Forwards a tensor through the Autoencoder and the discriminator

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input images

        Returns
        -------
        dict
            a dictionary containing the network's outputs

        """
        results_gen = self.generator(x)

        z = torch.randn_like(results_gen["encoded"])

        return {**results_gen,
                "discr_encoded": self.discriminator(results_gen["encoded"]),
                "discr_noise": self.discriminator(z)}


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

    predictions = model(data_dict["data"])

    adv_loss_gen = losses["adversarial"](predictions["discr_encoded"],
                                         True)
    pixel_loss_gen = losses["pixelwise"](predictions["decoded"],
                                         data_dict["data"])

    total_loss_gen = 0.001 * adv_loss_gen + 0.999 * pixel_loss_gen

    optimizers["generator"].zero_grad()
    total_loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    real_loss = losses["adversarial"](predictions["discr_noise"], True)
    fake_loss = losses["adversarial"](predictions["discr_encoded"], False)

    total_loss_discr = 0.5 * (real_loss + fake_loss)

    optimizers["discriminator"].zero_grad()
    total_loss_discr.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
