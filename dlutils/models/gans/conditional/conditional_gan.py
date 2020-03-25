import torch

from dlutils.models.gans.conditional.models import Discriminator, \
    Generator


class ConditionalGAN(torch.nn.Module):
    """
    An implementation of conditional generative adversarial networks, which
    are capable of generating samples of specific classes by providing the
    class label.

    References
    ----------
    `Paper <https://arxiv.org/abs/1411.1784>`_

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

    def __init__(self, latent_dim, n_classes, img_shape,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        latent_dim : int
            the size of the latent space
        n_classes : int
            the total number of classes
        img_shape : tuple
            the shape of the image batch (including channel dimension,
            excluding batch dimension)
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """
        super().__init__()

        self.generator = generator_cls(latent_dim=latent_dim,
                                       n_classes=n_classes,
                                       img_shape=img_shape)
        self.discriminator = discriminator_cls(n_classes=n_classes,
                                               img_shape=img_shape)
        self._latent_dim = latent_dim
        self._n_classes = n_classes

    def forward(self, x: torch.Tensor, labels: torch.Tensor,
                z: torch.Tensor = None,
                gen_labels: torch.Tensor = None):
        """
        Forwards inputs (and intermediate outputs) through all necessary neural
        networks.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the original image batch
        labels : :class:`torch.Tensor`
            the original label batch
        z : :class:`torch.Tensor`
            a noise batch; Will be sampled from normal distribution
            if not given
        gen_labels : :class:`torch.Tensor`
            a batch of class labels, the generated images should contain;
            Will be sampled from random number generator if not given

        Returns
        -------
        dict
            a dictionary containing all oututs necessary for training and loss
            calculation

        """

        if z is None:
            z = torch.randn(x.size(0), self._latent_dim, device=x.device,
                            dtype=x.dtype)

        if gen_labels is None:
            gen_labels = torch.randint_like(labels, 0, self._n_classes)

        gen_imgs = self.generator(z, gen_labels)

        validity_fake = self.discriminator(gen_imgs, gen_labels)
        validity_real = self.discriminator(x, labels)

        return {
            "gen_imgs": gen_imgs, "validity_fake": validity_fake,
            "validity_real": validity_real
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

    preds = model(x=data_dict["data"], labels=data_dict["label"])

    loss_gen = losses["adversarial"](preds["validity_fake"], True)
    loss_discr_fake = losses["adversarial"](preds["validity_fake"], False)
    loss_discr_real = losses["adversarial"](preds["validity_real"], True)
    loss_discr = (loss_discr_real + loss_discr_fake) / 2

    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    optimizers["discriminator"].zero_grad()
    loss_discr.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
