import torch

from dlutils.models.gans.auxiliary_classifier.models import \
    Generator, Discriminator
from dlutils.models.gans.utils import weights_init_normal


class AuxiliaryClassifierGANPyTorch(torch.nn.Module):
    """
    A Generative Adversarial Network employing classification as auxiliary task

    References
    ----------
    `Paper <https://arxiv.org/abs/1610.09585>`_

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

    def __init__(self, n_classes=10, latent_dim=100, img_size=32,
                 n_channels=1, generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        n_classes : int
            the number of classes
        latent_dim : int
            the size of the latent dimension
        img_size : int
            the size of the squared images (per dimension)
        n_channels : int
            the number of image channels
        generator_cls :
            a subclass of :class:`torch.nn.Module implementing the actual
            generator model
        discriminator_cls :
            a subclass of :class:`torch.nn.Module implementing the actual
            discriminator model

        """
        super().__init__()
        self._n_classes = n_classes
        self._n_channels = n_channels
        self._latent_dim = latent_dim
        self._img_size = img_size
        self.generator = generator_cls(n_classes=n_classes,
                                       latent_dim=latent_dim,
                                       img_size=img_size,
                                       n_channels=n_channels)
        self.discriminator = discriminator_cls(n_classes=n_classes,
                                               img_size=img_size,
                                               n_channels=n_channels)

        self.reset_parameters()

    def forward(self, real_imgs: torch.Tensor = None,
                real_labels: torch.Tensor = None,
                noise: torch.Tensor = None,
                gen_labels: torch.Tensor = None):
        """
        Forwards all tensors through the corresponding networks

        Parameters
        ----------
        real_imgs : :class:`torch.Tensor`
            the batch of real images
        real_labels : :class:`torch.Tensor`
            the batch of real labels
        noise : :class:`torch.Tensor`
            the noise vector; will be generated if not provided
        gen_labels : :class:`torch.Tensor`
            the tensor of artificial class labels; Will be generated if not
            provided

        Returns
        -------
        dict
            a dictionary containing all obtained predictions

        Warnings
        --------
        At least one combination of  :param:`real_imgs` and
        :param:`real_labels` or  :param:`noise` and :param:`gen_labels` must
        be given, otherwise the predictions won't work

        Notes
        -----
        If :param:`noise` or :param:`gen_labels` are not given, but the
        :param:´real_imgs` and :param:`real_labels` are, the missing tensors
        are sampled from normal distributions.

        If :param`real_imgs` is not given, the corresponding predictions will
        be missing in the result dict

        """

        # at least one of the combinations of
        # [(real_imgs, real_labels), (noise, gen_labels)] must be given
        assert (real_imgs is not None and real_labels is not None) or \
               (noise is not None and gen_labels is not None), \
            "Either real images or noise must be provided"

        if noise is None:
            noise = torch.randn(real_imgs.size(0), self._latent_dim,
                                device=real_imgs.device, dtype=real_imgs.dtype)

        if gen_labels is None:
            gen_labels = torch.randint(0, self._n_classes,
                                       (real_imgs.size(0),),
                                       device=real_imgs.device)

        gen_imgs = self.generator(noise, gen_labels)
        validity_gen, pred_label_gen = self.discriminator(gen_imgs)

        result_dict = {
            "gen_labels": gen_labels,
            "gen_imgs": gen_imgs,
            "validity_gen": validity_gen,
            "pred_label_gen": pred_label_gen
        }

        if real_imgs is not None:
            real_pred, real_aux = self.discriminator(real_imgs)
            result_dict["real_pred"] = real_pred
            result_dict["real_aux"] = real_aux

        return result_dict

    def reset_parameters(self):
        """
        Function to reset the parameters of generator and discriminator

        """
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)


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
        Metric values (with same keys as input dict metrics)
    dict
        Loss values (with same keys as input dict losses)
    dict
        Arbitrary number of predictions

    """

    predictions = model(real_imgs=data_dict["data"],
                        real_labels=data_dict["label"])

    # calculate generator losses
    adv_loss_gen = losses["adversarial"](predictions["validity_gen"], True)
    aux_loss_gen = losses["auxiliary"](predictions["pred_label_gen"],
                                       predictions["gen_labels"])

    gen_loss = 0.5 * (adv_loss_gen + aux_loss_gen)

    # update generator
    optimizers["generator"].zero_grad()
    # retain graph for discriminator update
    gen_loss.backward(retain_graph=True)
    optimizers["generator"].step()

    # calculate discriminator losses
    adv_loss_discr = losses["adversarial"](predictions["real_pred"], False)
    aux_loss_discr = losses["auxiliary"](predictions["pred_label_gen"],
                                         predictions["gen_labels"])

    discr_loss = 0.5 * (adv_loss_discr + aux_loss_discr)

    # update discriminator
    optimizers["discriminator"].zero_grad()
    discr_loss.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
