import torch

from dlutils.models.gans.semi_supervised.models import Discriminator, \
    Generator


class SemiSupervisedGAN(torch.nn.Module):
    """
    Class implementing Semi-Supervised Learning with Generative Adversarial
    Networks

    References
    ----------
    `Paper <https://arxiv.org/abs/1606.01583>`_

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

    def __init__(self, latent_dim, img_size, num_channels, num_classes=10,
                 generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        img_size : int
            number of pixels per side of the image
        num_channels : int
            number of image channels
        num_classes : int
            number of image classes
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """

        super().__init__()

        self.generator = generator_cls(latent_dim, img_size, num_channels,
                                       num_classes)

        self.discriminator = discriminator_cls(img_size, num_channels,
                                               num_classes)

        self._latent_dim = latent_dim
        self._n_classes = num_classes

    def forward(self, x, z=None):
        """
        Feeds a single set of batches through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        z : :class:`torch.Tensor`

        Returns
        -------
        dict
            a dictionary containing all the (intermediate) results necessary
            for loss calculation and training

        """

        if z is None:
            z = torch.randn(x.size(0), self._latent_dim, device=x.device,
                            dtype=x.dtype)

        gen_imgs = self.generator(z)

        val_fake, labels_fake = self.discriminator(gen_imgs)
        val_real, labels_real = self.discriminator(x)

        fake_gt_label = torch.randint(self._n_classes, x.size(0),
                                      device=gen_imgs.device)

        return {"gen_imgs": gen_imgs, "val_fake": val_fake,
                "labels_fake": labels_fake, "val_real": val_real,
                "labels_real": labels_real, "fake_gt_label": fake_gt_label}


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

    gen_loss = losses["adversarial"](preds["val_fake"], True)

    optimizers["generator"].zero_grad()
    gen_loss.backward(retain_graph=True)
    optimizers["generator"].step()

    discr_fake_adv = losses["adversarial"](preds["val_fake"], False)
    discr_fake_aux = losses["auxiliary"](preds["labels_fake"],
                                         preds["fake_gt_label"])

    discr_fake = (discr_fake_adv + discr_fake_aux) / 2

    discr_real_adv = losses["adversarial"](preds["val_real"], True)
    discr_real_aux = losses["auxiliary"](preds["labels_real"],
                                         data_dict["label"])

    discr_real = (discr_real_adv + discr_real_aux) / 2

    discr_loss = (discr_fake + discr_real) / 2

    optimizers["discriminator"].zero_grad()
    discr_loss.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
