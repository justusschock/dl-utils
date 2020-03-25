import torch

from dlutils.models.gans.deep_convolutional.models import Generator, \
    Discriminator
from dlutils.models.gans.utils import weights_init_normal


class DeepConvolutionalGAN(torch.nn.Module):
    """
    Implementation of Deep Convolutional Generative Adversarial Networks for
    image synthesis with exchangeable generator and discriminator classes

    References
    ----------
    `Paper <https://arxiv.org/abs/1511.06434>`_

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

    def __init__(self, latent_dim, img_size, num_channels,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        super().__init__()

        self.generator = generator_cls(latent_dim, img_size, num_channels)
        self.discriminator = discriminator_cls(num_channels, img_size)

        self._latent_dim = latent_dim
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    def forward(self, imgs: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn(imgs.size(0), self._latent_dim,
                                device=imgs.device, dtype=imgs.dtype)

        gen_imgs = self.generator(noise)

        discr_real = self.discriminator(imgs)
        discr_fake = self.discriminator(gen_imgs)

        return {"gen_imgs": gen_imgs, "discr_real": discr_real,
                "discr_fake": discr_fake}


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

    gen_loss = losses["adversarial"](preds["discr_fake"], True)

    optimizers["generator"].zero_grad()
    gen_loss.backward(retain_graph=True)
    optimizers["generator"].step()

    real_loss = losses["adversarial"](preds["discr_real"], True)
    fake_loss = losses["adversarial"](preds["discr_fake"], False)

    discr_loss = (fake_loss + real_loss) / 2

    optimizers["discriminator"].zero_grad()
    discr_loss.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
