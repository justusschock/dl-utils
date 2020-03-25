import torch

from dlutils.models.gans.super_resolution.models import GeneratorResNet, \
    Discriminator, FeatureExtractor


class SuperResolutionGAN(torch.nn.Module):
    """
    Class implementing Photo-Realistic Single Image Super-Resolution
    Using a Generative Adversarial Network

    References
    ----------
    `Paper <https://arxiv.org/abs/1609.04802>`_

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

    def __init__(self, img_shape, num_residuals, generator_cls=GeneratorResNet,
                 feature_extractor_cls=FeatureExtractor,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the input/generated images (including channels,
            excluding batch dimension)
        num_residuals : int
            number of residual blocks inside the generator
        generator_cls :
            class implementing the actual generator topology
        feature_extractor_cls :
            class implementing the actual feature extractor topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """

        super().__init__()

        self.generator = generator_cls(img_shape[0], img_shape[0],
                                       num_residuals)
        self.feature_extractor = feature_extractor_cls()
        self.discriminator = discriminator_cls(img_shape[0])

    def forward(self, imgs_lr, imgs_hr):
        """
        Feeds a single set of batches through the networks

        Parameters
        ----------
        imgs_lr : :class:`torch.Tensor`
            the low resolution images
        imgs_hr : :class:`torch.Tensor`
            the high resolution images

        Returns
        -------
        dict
            a dictionary containing all the necessary (intermediate) outputs
            for loss calculation and training

        """

        gen_hr = self.generator(imgs_lr)

        gen_features = self.feature_extractor(gen_hr)
        real_features = self.feature_extractor(imgs_hr)

        discr_real = self.discriminator(imgs_hr)
        discr_fake = self.discriminator(gen_hr)

        return {"gen_hr": gen_hr, "gen_features": gen_features,
                "real_features": real_features, "discr_real": discr_real,
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

    preds = model(data_dict["data_lr"], data_dict["data_hr"])

    gen_adversarial = losses["adversarial"](preds["discr_fake"], True)
    gen_content = losses["content"](preds["gen_features"],
                                    preds["real_features"])

    loss_gen = gen_content + 1e-3 * gen_adversarial

    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    discr_real = losses["adversarial"](preds["discr_real"], True)
    discr_fake = losses["adversarial"](preds["discr_fake"], False)

    loss_discr = (discr_real + discr_fake) / 2

    optimizers["discriminator"].zero_grad()
    loss_gen.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
