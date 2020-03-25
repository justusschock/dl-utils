import torch

from dlutils.models.gans.pixel_da.models import Generator, \
    Discriminator, Classifier
from dlutils.models.gans.utils import weights_init_normal


class PixelDomainAdaptation(torch.nn.Module):
    """
    Class implementing the Unsupervised Pixel-Level Domain Adaptation with
    Generative Adversarial Networks

    References
    ----------
    `Paper <https://arxiv.org/abs/1612.05424>`_

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

    def __init__(self, img_size, num_channels=1, latent_dim=100, n_classes=10,
                 n_residual_blocks=6, num_filts=64, lambda_task=0.1,
                 lambda_adv=1., generator_cls=Generator,
                 discriminator_cls=Discriminator, clf_cls=Classifier):
        """

        Parameters
        ----------
        img_size : int
            number of pixels per image side
        num_channels : int
            number of image channels
        latent_dim : int
            size of latent dimension
        n_classes : int
            number of classes
        n_residual_blocks : int
            number of residual blocks to include
        num_filts : int
            number of filters per convolution
        lambda_task : float
            weighting of task loss
        lambda_adv : float
            weighting of adversarial loss
        generator_cls :
            class implementing the actual generator class
        discriminator_cls :
            class implementing the actual discriminator class
        clf_cls :
            class implementing the actual classifier class

        """

        super().__init__()

        self.generator = generator_cls(latent_dim, num_channels, img_size,
                                       n_residual_blocks, num_filts)
        self.discriminator = discriminator_cls(num_channels)
        self.classifier = clf_cls(num_channels, img_size, n_classes)

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.classifier.apply(weights_init_normal)

        self._latent_dim = latent_dim
        self.lambda_task = lambda_task
        self.lambda_adversarial = lambda_adv

    def forward(self, imgs_a, imgs_b, noise=None):
        """
        Feeds a set of batches through the necessary networks

        Parameters
        ----------
        imgs_a : :class:`torch.Tensor`
            batch of images of domain A (source domain)
        imgs_b : :class:`torch.Tensor`
            batch of images of domain B (target domain)
        noise : :class:`torch.Tensor`
            batch of noise vectors; will be sampled if not given

        Returns
        -------
        dict
            dictionary containing all (intermediate) outputs for training and
            loss calculation

        """

        if noise is None:
            noise = torch.empty(imgs_a.size(0), self._latent_dim,
                                device=imgs_a.device, dtype=imgs_a.dtype)

            noise.uniform_(-1, 1)

        fake_b = self.generator(imgs_a, noise)
        label_pred_fake = self.classifier(fake_b)
        labels_pred_real_a = self.classifier(imgs_a)
        labels_pred_real_b = self.classifier(imgs_b)

        discr_fake = self.discriminator(fake_b)
        discr_real = self.discriminator(imgs_b)

        return {"gen_imgs": fake_b, "label_pred_fake": label_pred_fake,
                "labels_pred_real_a": labels_pred_real_a,
                "labels_pred_real_b": labels_pred_real_b,
                "discr_fake": discr_fake, "discr_real": discr_real}


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
        lambda_task = model.module.lambda_task
        lambda_adv = model.module.lambda_adversarial
    else:
        lambda_task = model.lambda_task
        lambda_adv = model.lambda_adversarial

    preds = model(data_dict["data_a"], data_dict["data_b"])

    gen_task_real = losses["task"](preds["labels_pred_real_a"],
                                   data_dict["label_a"])
    gen_task_fake = losses["task"](preds["labels_pred_fake"],
                                   data_dict["label_a"])

    gen_task_loss = (gen_task_real + gen_task_fake) / 2
    adversarial = losses["adversarial"](preds["discr_fake"], True)

    loss_gen = lambda_task * gen_task_loss + lambda_adv * adversarial

    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    real_loss = losses["adversarial"](preds["discr_real"], True)
    fake_loss = losses["adversarial"](preds["discr_fake"], False)

    loss_discr = (real_loss + fake_loss) / 2

    optimizers["discriminator"].zero_grad()
    loss_discr.backward()
    optimizers["discriminator"].step()

    metric_vals["acc_translated"] = torch.mean((torch.argmax(
        preds["labels_pred_fake"]) == data_dict["label_a"]
    ).to(torch.float))

    metric_vals["acc_translated"] = torch.mean((torch.argmax(
        preds["labels_pred_real_b"]) == data_dict["label_b"]
    ).to(torch.float))

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
