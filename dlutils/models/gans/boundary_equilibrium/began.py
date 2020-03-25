import torch

from dlutils.models.gans.boundary_equilibrium.models import \
    Generator, Discriminator
from dlutils.models.gans.utils import weights_init_normal


class BoundaryEquilibriumGAN(torch.nn.Module):
    """
    A basic implementation of Boundary Equilibrium Generative Adversarial
    Networks with variable generator and discriminator networks

    This GAN is trained with a loss derived from the Wasserstein disctance
    enforcing a equilibrium of generator and discriminator during training.


    References
    ----------
    `Paper <https://arxiv.org/abs/1703.10717>`_

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

    def __init__(self, n_channels, latent_dim, img_size,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        n_channels : int
            the number of image channels
        latent_dim : int
            the size of the latent dimension
        img_size : int
            the size of the squared input images (per side)
        generator_cls :
            subclass of :class:`torch.nn.Module implementing the actual
            generator topology
        discriminator_cls :
            subclass of :class:`torch.nn.Module implementing the actual
            discriminator topology

        """
        super().__init__()
        self.generator = generator_cls(n_channels=n_channels,
                                       latent_dim=latent_dim,
                                       img_size=img_size)

        self.discriminator = discriminator_cls(n_channels=n_channels,
                                               img_size=img_size)

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self._latent_dim = latent_dim

    def forward(self, x: torch.Tensor, noise=None):
        """
        Forwards a real image batch and an image batch generated from (sampled)
        noise through the discriminator

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the batch of real images
        noise : :class:`torch.Tensor`
            the noise vector to generate images from;
            If None: noise vector will be sampled from normal distrbution

        Returns
        -------
        dict
            a dictionary containing all relevant predictions

        """
        if noise is None:
            noise = torch.randn(x.size(0),
                                self._latent_dim, device=x.device,
                                dtype=x.dtype)

        gen_imgs = self.generator(noise)

        return {"gen_imgs": gen_imgs, "discr_real": self.discriminator(x),
                "discr_fake": self.discriminator(gen_imgs)}


def update_fn(model, data_dict: dict, optimizers: dict, losses=None,
              ):
    """
    Function which handles prediction from batch, logging, loss calculation
    and optimizer step

    Parameters
    ----------
    model : :class:` torch.nn.Module
        model to forward data through
    data_dict : dict
        dictionary containing the data
    optimizers : dict
        dictionary containing all optimizers to perform parameter update
    losses : dict
        Functions or classes to calculate losses


    Returns
    -------
    dict
        Metric values (with same keys as input dict metrics);
        Will always be empty here
    dict
        Loss values (with same keys as input dict losses)
    dict
        Arbitrary number of predictions

    """

    predictions = model(data_dict["data"])

    loss_gen = (predictions["discr_fake"] - data_dict["data"]).abs().mean()

    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    discr_loss, discr_loss_real, discr_loss_fake = losses["began"](
        predictions["discr_real"],
        data_dict["data"],
        predictions["discr_fake"],
        predictions["gen_imgs"])

    optimizers["discriminator"].zero_grad()
    discr_loss.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
