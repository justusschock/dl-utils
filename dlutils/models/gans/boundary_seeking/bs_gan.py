import torch

from dlutils.models.gans.boundary_seeking.models import Generator, \
    Discriminator


class BoundarySeekingGAN(torch.nn.Module):
    """
    A Basic Implementation of the boundary seeking generative adversarial
    networks with variable generator and discriminator class

    This GAN derivate is trained by applying policy gradients to the generator
    to make it working with non-differentiable, discrete data. It also
    improves stability of training in continuous data.

    References
    ----------
    `Paper <https://arxiv.org/abs/1702.08431>`_

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
            the size of the latent dimension
        img_shape : tuple
            the shape of image batches (including channel-dimension, excluding
            batch-dimension)
        generator_cls :
            the class implementing the actual generator topology
        discriminator_cls :
            the class implementing the actual discriminator topology

        """
        super().__init__()
        self.generator = generator_cls(latent_dim, img_shape)
        self.discriminator = discriminator_cls(img_shape)
        self._latent_dim = latent_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        if z is None:
            z = torch.randn(x.size(0), self._latent_dim, device=x.device,
                            dtype=x.dtype)

        # ToDo: Generator Parameters become NaN at some time in first epoch,
        #  when training with MNIST, but loss doesn't
        gen_imgs = self.generator(z)

        discr_real = self.discriminator(x)
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
    model : :class:` torch.nn.Module
        model to forward data through
    data_dict : dict
        dictionary containing the data
    optimizers : dict
        dictionary containing all optimizers to perform parameter update
    losses : dict
        Functions or classes to calculate losses
    metrics : dict
        Functions or classes to calculate other metrics; won't be used here
    fold : int
        Current Fold in Crossvalidation (default: 0)
    kwargs : dict
        additional keyword arguments

    Returns
    -------
    dict
        Metric values (with same keys as input dict metrics)
    dict
        Loss values (with same keys as input dict losses)
    dict
        Arbitrary number of predictions

    """

    preds = model(data_dict["data"])

    loss_gen = losses["boundary_seeking"](preds["discr_fake"])

    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    real_loss = losses["adversarial"](preds["discr_real"], True)
    fake_loss = losses["adversarial"](preds["discr_fake"], True)

    discr_loss = 0.5 * (real_loss + fake_loss)

    optimizers["discriminator"].zero_grad()
    discr_loss.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
