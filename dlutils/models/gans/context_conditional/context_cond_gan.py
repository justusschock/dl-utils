import torch

from dlutils.models.gans.context_conditional.models import Discriminator, \
    Generator


class ContextConditionalGAN(torch.nn.Module):
    """
    Skeleton for Context Conditional Generative Adversarial Networks with
    exchangeable Generator and Discriminator classes

    This GAN version is suitable for in-painting using a semi-supervised
    training approach.

    References
    ----------
    `Paper <https://arxiv.org/abs/1611.06430>`_

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

    def __init__(self, img_shape, mask_size, generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the image batches (including channels, excluding
            batch dimension)
        mask_size : int
            the size of the mask
        generator_cls :
            the class implementing the actual generator topology
        discriminator_cls :
            the class implementing the actual discriminator topology

        """
        super().__init__()

        self.generator = generator_cls(img_shape[0])
        self.discriminator = discriminator_cls(img_shape)
        self._img_size = img_shape[1]
        self._mask_size = mask_size

    @staticmethod
    def apply_mask(imgs: torch.Tensor, mask: torch.Tensor, mask_size: int):
        """

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            the image batch
        mask : :class:`torch.Tensor`
            the mask to apply
        mask_size : int
            the size of the mask (needed for index computation)

        Returns
        -------
        :class:`torch.Tensor`
            the masked image batch

        """
        masked_imgs = imgs.clone()
        for i, (y1, x1) in enumerate(mask):
            y2, x2 = y1 + mask_size, x1 + mask_size
            masked_imgs[i, :, y1:y2, x1:x2] = 1

        return masked_imgs

    @staticmethod
    def _generate_random_mask(batchsize, max_val):
        """
        Generates a random mask

        Parameters
        ----------
        batchsize : int
            the number of masks to generate
        max_val : int
            the maximum index value for a mask

        Returns
        -------
        :class:`torch.Tensor`
            the indices to mask

        """
        return torch.randint(max_val, (batchsize, 2))

    def apply_random_mask(self, imgs):
        """
        Generates a random mask and directly applies it on a given image batch

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the masked image batch

        """
        mask = self._generate_random_mask(imgs.size(0),
                                          self._img_size - self._mask_size)

        return self.apply_mask(imgs, mask, self._mask_size)

    def forward(self, real_imgs, imgs_lr, masks: torch.Tensor = None):
        """
        Forwards a given set of batches through the networks

        Parameters
        ----------
        real_imgs : :class:`torch.Tensor`
            the real images
        imgs_lr : :class:`torch.Tensor`
            the images to use for the inpainting
        masks : :class:`torch.Tensor
            the mask specifiying where to apply inpainting; must be of shape
            (N x 2 )

        Returns
        -------
        dict
            a dictionary of :class:`torch.Tensor` containing all returned
            values

        """
        if masks is None:
            masked_imgs = self.apply_random_mask(real_imgs)
        else:
            masked_imgs = self.apply_mask(real_imgs, masks, self._mask_size)

        gen_imgs = self.generator(masked_imgs, imgs_lr)

        discr_real = self.discriminator(real_imgs)
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

    preds = model(data_dict["data"], data_dict["data_lr"])

    loss_gen = losses["adversarial"](preds["discr_fake"], True)

    optimizers["generator"].zero_grad()
    loss_gen.backward(retain_graph=True)
    optimizers["generator"].step()

    loss_fake = losses["adversarial"](preds["discr_fake"], False)
    loss_real = losses["adversarial"](preds["discr_real"], True)

    loss_discr = (loss_fake + loss_real) / 2

    optimizers["discriminator"].zero_grad()
    loss_discr.backward()
    optimizers["discriminator"].step()

    # zero gradients again just to make sure, gradients aren't carried to
    # next iteration (won't affect training since gradients are zeroed
    # before every backprop step, but would result in way higher memory
    # consumption)
    for k, v in optimizers.items():
        v.zero_grad()
