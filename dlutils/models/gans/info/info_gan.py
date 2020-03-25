import torch
from rising.transforms.functional.channel import one_hot_batch

from dlutils.models.gans.info.models import Discriminator, Generator
from dlutils.models.gans.utils import weights_init_normal


class InfoGAN(torch.nn.Module):
    """
    Class implementing the
    Information Maximization Generative Adversarial Networks.

    References
    ----------
    `Paper <https://arxiv.org/abs/1606.03657>`_

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

    def __init__(self, latent_dim, n_classes, code_dim, img_size, num_channels,
                 lambda_cat=1., lambda_cont=0.1,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension
        n_classes : int
            the number of classes
        code_dim : int
            the size of the code dimension
        img_size : int
            the number of pixels per image side
        num_channels : int
            number of image channels
        lambda_cat : float
            weighting factor specifying the impact of the categorical loss
            onto the info-loss
        lambda_cont : float
            weighting factor specifying the impact of the continuous loss onto
            the info loss
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """
        super().__init__()

        self.generator = generator_cls(latent_dim, n_classes, code_dim,
                                       img_size, num_channels)

        self.discriminator = discriminator_cls(code_dim, n_classes,
                                               num_channels, img_size)

        self._n_classes = n_classes
        self._latent_dim = latent_dim
        self._code_dim = code_dim

        self.lambda_cat = lambda_cat
        self.lambda_cont = lambda_cont

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    def forward(self, imgs, labels=None, z=None, code=None):
        """
        Forwards a single set of batches through the network

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            the image batch
        labels : :class:`torch.Tensor`
            the labels batch, will be sampled if not given
        z : :class:`torch.Tensor`
            the noise batch, will be sampled if not given
        code : :class:`torch.Tensor`
            the code batch, will be sampled if not given

        Returns
        -------
        dict
            a dictionary containing all (intermediate) results for loss
            calculation and training

        """

        if z is None:
            z = torch.randn(imgs.size(0), self._latent_dim, device=imgs.device,
                            dtype=imgs.dtype)

        if labels is None:
            labels = torch.randint(self._n_classes,
                                   (imgs.size(0), 1),
                                   device=imgs.device,
                                   dtype=torch.long)

        if labels.size(-1) != self._n_classes:
            labels = one_hot_batch(labels.unsqueeze(1),
                                   num_classes=self._n_classes)

        if code is None:
            code = torch.empty(imgs.size(0), self._code_dim,
                               device=imgs.device, dtype=imgs.dtype)
            code.uniform_(-1, 1)

        gen_imgs = self.generator(z, labels, code)

        validity_real, _, _ = self.discriminator(imgs)
        validity_fake, labels_fake, code_fake = self.discriminator(gen_imgs)

        return {
            "validity_real": validity_real, "validity_fake": validity_fake,
            "labels_real": labels, "labels_fake": labels_fake,
            "code_real": code, "code_fake": code_fake,
            "gen_imgs": gen_imgs,
        }


def update_fn(model, data_dict: dict, optimizers: dict, losses={},
              metrics={}, fold=0, **kwargs):
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

    preds = model(data_dict["data"], data_dict["label"])

    # calculate generator loss
    g_loss = losses["adversarial"](preds["validity_fake"], True)

    # update generator
    optimizers["generator"].zero_grad()
    g_loss.backward(retain_graph=True)
    optimizers["generator"].step()

    # calculate discriminator losses
    loss_real = losses["adversarial"](preds["validity_real"], True)
    loss_fake = losses["adversarial"](preds["validity_fake"], False)

    loss_discr = (loss_real + loss_fake) / 2

    # update discriminator
    optimizers["discriminator"].zero_grad()
    loss_discr.backward(retain_graph=True)
    optimizers["discriminator"].step()

    # calculate info losses
    loss_categorical = losses["categorical"](
        preds["labels_fake"],
        torch.argmax(preds["labels_real"], dim=-1))

    loss_continuous = losses["continuous"](preds["code_fake"],
                                           preds["code_real"])

    loss_info = (attr_module.lambda_cat * loss_categorical
                 + attr_module.lambda_cont * loss_continuous)

    optimizers["info"].zero_grad()
    loss_info.backward()
    optimizers["info"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
