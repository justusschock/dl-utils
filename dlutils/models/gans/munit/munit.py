import torch

from dlutils.models.gans.munit.models import Encoder, Decoder, \
    MultiResolutionDiscriminator


class MUNIT(torch.nn.Module):
    """
    Class implementing the
    Multimodal Unsupervised Image-to-Image Translation

    References
    ----------
    `Paper <https://arxiv.org/abs/1804.04732>`_

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

    def __init__(self, num_channels_a=3, num_channels_b=3, n_filts=64,
                 n_residual=3, n_sample=2,
                 style_dim=8, num_discr_paths=3,
                 lambda_gan=1., lambda_identity=10., lambda_style=1.,
                 lambda_content=1., lambda_cycle=0.,
                 encoder_cls=Encoder, decoder_cls=Decoder,
                 discriminator_cls=MultiResolutionDiscriminator):
        """

        Parameters
        ----------
        num_channels_a : int
            number of image channels for domain A
        num_channels_b : int
            number of image channels for domain B
        n_filts : int
            number of convolutional filters per layer
        n_residual : int
            number of residual blocks in Encoders and Decoders
        n_sample : int
            number of up-/downsampling blocks in Encoders and Decoders
        style_dim : int
            size of the latent dimension for style encodings
        num_discr_paths : int
            number of parallel discriminator paths (each with a lower
            resolution than the ones before)
        lambda_gan : float
            weighting factor for the adversarial loss
        lambda_identity : float
            weighting factor for the identity (reconstruction) loss
        lambda_style : float
            weighting factor for the style loss
        lambda_content : float
            weighting factor for the content loss
        lambda_cycle : float
            weighting factor for the cycle-consistency loss
        encoder_cls :
            class implementing the actual encoder topology
        decoder_cls :
            class implementing the actual decoder topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """
        super().__init__()

        # encodes images of domain A
        self.encoder_a = encoder_cls(num_channels_a, n_filts, n_residual,
                                     n_sample, style_dim)
        # encodes images of domain B
        self.encoder_b = encoder_cls(num_channels_b, n_filts, n_residual,
                                     n_sample, style_dim)

        # decodes images of domain A
        self.decoder_a = decoder_cls(num_channels_a, n_filts, n_residual,
                                     n_sample, style_dim)
        # decodes images of domain B
        self.decoder_b = decoder_cls(num_channels_b, n_filts, n_residual,
                                     n_sample, style_dim)

        # discriminator working on images of domain A
        self.discr_a = discriminator_cls(num_channels_a, num_discr_paths)
        self.discr_b = discriminator_cls(num_channels_b, num_discr_paths)

        self._style_dim = style_dim

        self.lambda_adversarial = lambda_gan
        self.lambda_identity = lambda_identity
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.lambda_cycle = lambda_cycle

    def forward(self, imgs_a, imgs_b):
        """
        Feeds a batch of each domain through all networks

        Parameters
        ----------
        imgs_a : :class:`torch.Tensor`
            the images of domain A
        imgs_b : :class:`torch.Tensor`
            the images of domain B

        Returns
        -------
        dict
            dictionary containing all (intermediate) results for loss
            calculation and training

        """

        # get encodings for content and style
        cont_code_a, style_code_a = self.encoder_a(imgs_a)
        cont_code_b, style_code_b = self.encoder_b(imgs_b)

        # reconstruct images
        rec_a = self.decoder_a(cont_code_a, style_code_a)
        rec_b = self.decoder_b(cont_code_b, style_code_b)

        # translate images
        fake_a = self.decoder_a(cont_code_b, style_code_a)
        fake_b = self.decoder_b(cont_code_a, style_code_b)

        # cyclic translation
        cont_code_fake_a, style_code_fake_a = self.encoder_a(fake_a)
        cont_code_fake_b, style_code_fake_b = self.encoder_b(fake_b)

        fake_rec_a = self.decoder_a(cont_code_fake_b, style_code_a)
        fake_rec_b = self.decoder_b(cont_code_fake_a, style_code_b)

        # discriminator results
        discr_fake_a = self.discr_a(fake_a)
        discr_fake_b = self.discr_b(fake_b)
        discr_real_a = self.discr_a(imgs_a)
        discr_real_b = self.discr_b(imgs_b)

        # sample from normal distribution
        sampled_style_a = torch.randn(imgs_a.size(0), self._style_dim,
                                      device=imgs_a.device, dtype=imgs_a.dtype)
        sampled_style_b = torch.randn(imgs_b.size(0), self._style_dim,
                                      device=imgs_b.device, dtype=imgs_b.dtype)

        return {
            "cont_code_a": cont_code_a, "style_code_a": style_code_a,
            "cont_code_b": cont_code_b, "style_code_b": style_code_b,
            "rec_a": rec_a, "rec_b": rec_b, "fake_a": fake_a, "fake_b": fake_b,
            "cont_code_fake_a": cont_code_fake_a, "fake_rec_a": fake_rec_a,
            "cont_code_fake_b": cont_code_fake_b, "fake_rec_b": fake_rec_b,
            "style_code_fake_a": style_code_fake_a,
            "style_code_fake_b": style_code_fake_b,
            "discr_fake_a": discr_fake_a, "discr_fake_b": discr_fake_b,
            "sampled_style_a": sampled_style_a, "discr_real_a": discr_real_a,
            "sampled_style_b": sampled_style_b, "discr_real_b": discr_real_b
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

    if isinstance(model, torch.nn.DataParallel):
        attr_module = model.module
    else:
        attr_module = model

    preds = model(data_dict["data_a"], data_dict["data_b"])

    # Adversarial Losses
    loss_gan_a = losses["multiresolution_adversarial"](
        preds["discr_fake_a"], True)
    loss_gan_b = losses["multiresolution_adversarial"](
        preds["discr_fake_b"], True)

    # Identity Losses
    loss_id_a = losses["reconstruction"](preds["rec_a"],
                                         data_dict["data_a"])
    loss_id_b = losses["reconstruction"](preds["rec_b"],
                                         data_dict["data_b"])

    # Style Loss
    loss_style_a = losses["reconstruction"](preds["style_code_fake_a"],
                                            preds["sampled_style_a"])
    loss_style_b = losses["reconstruction"](preds["style_code_fake_b"],
                                            preds["sampled_style_b"])

    # Content Loss
    loss_content_a = losses["reconstruction"](preds["cont_code_fake_b"],
                                              preds["cont_code_a"].detach()
                                              )
    loss_content_b = losses["reconstruction"](preds["cont_code_fake_a"],
                                              preds["cont_code_b"].detach()
                                              )

    # Cyclic Reconstruction Loss
    loss_cyc_a = losses["reconstruction"](preds["fake_rec_a"],
                                          data_dict["imgs_a"])
    loss_cyc_b = losses["reconstruction"](preds["fake_rec_b"],
                                          data_dict["imgs_b"])

    # calculate total loss for generative part
    loss_gen = (
            attr_module.lambda_adversarial * (loss_gan_a + loss_gan_b)
            + attr_module.lambda_identity * (loss_id_a + loss_id_b)
            + attr_module.lambda_style * (loss_style_a + loss_style_b)
            + attr_module.lambda_content * (loss_content_a + loss_content_b)
            + attr_module.lambda_cycle * (loss_cyc_a + loss_cyc_b)
    )

    # update generative models
    optimizers["generator"].step()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    # loss discriminator A
    loss_real_a = losses["multiresolution_adversarial"](
        preds["discr_real_a"], True)
    loss_fake_a = losses["multiresolution_adversarial"](
        preds["discr_fake_a"], False)

    loss_discr_a = loss_real_a + loss_fake_a

    # update discriminator A
    optimizers["discriminator_a"].zero_grad()
    loss_discr_a.backward()
    optimizers["discriminator_a"].step()

    # loss discriminator B
    loss_real_b = losses["multiresolution_adversarial"](
        preds["discr_real_b"], True)
    loss_fake_b = losses["multiresolution_adversarial"](
        preds["discr_fake_b"], False)

    loss_discr_b = loss_real_b + loss_fake_b

    # update discriminator B
    optimizers["discriminator_b"].zero_grad()
    loss_discr_b.backward()
    optimizers["discriminator_b"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
