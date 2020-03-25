import torch

from dlutils.models.gans.energy_based.models import Discriminator, \
    Generator


class EnergyBasedGAN(torch.nn.Module):
    """
    Class implementing the Energy-Based GANs and it's behavior
    during training.

    References
    ----------
    `Paper <https://arxiv.org/abs/1609.03126>`_

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

    def __init__(self, img_size, num_channels, latent_dim,
                 generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_size : int
            number of pixels per image side
        num_channels : int
            number of image channels
        latent_dim : int
            size of the latent dimension
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """

        super().__init__()

        self.generator = generator_cls(img_size, latent_dim, num_channels)
        self.discriminator = discriminator_cls(img_size, num_channels)

        self._latent_dim = latent_dim

    def forward(self, imgs: torch.Tensor, noise: torch.Tensor = None):
        """
        Forwards a batch of images and a noise vector through all networks

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            the image batch
        noise : :class:`torch.Tensor`
            the noise batch, will be sampled from random distribution if not
            given

        Returns
        -------
        dict
            a dictionary containing all (intermediate) outputs for
            loss calculation and training

        """
        if noise is None:
            noise = torch.randn(imgs.size(0), self._latent_dim,
                                dtype=imgs.dtype, device=imgs.device)

        gen_imgs = self.generator(noise)

        real_recon, real_embeddings = self.discriminator(imgs)
        fake_recon, fake_embeddings = self.discriminator(gen_imgs)

        return {
            "gen_imgs": gen_imgs,
            "real_recon": real_recon, "real_embeddings": real_embeddings,
            "fake_recon": fake_recon, "fake_embeddings": fake_embeddings
        }


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

    preds = model(data_dict["data"])

    pixelwise = losses["pixelwise"](preds["fake_recon"], preds["gen_imgs"])
    pullaway = losses["pullaway"](preds["fake_embeddings"])

    g_loss = pixelwise + pullaway

    optimizers["generator"].zero_grad()
    g_loss.backward(retain_graph=True)
    optimizers["generator"].step()

    discr_loss, discr_real, discr_fake = losses["discriminator_margin"](
        preds["real_recon"], data_dict["data"], preds["fake_recon"],
        preds["gen_imgs"]
    )

    optimizers["discriminator"].zero_grad()
    discr_loss.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
