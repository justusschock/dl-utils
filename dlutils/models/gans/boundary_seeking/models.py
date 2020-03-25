from functools import reduce
from operator import mul

import torch


class Generator(torch.nn.Module):
    """
    A simple Generator Model
    """

    def __init__(self, latent_dim, img_shape):
        """

        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension
        img_shape : tuple
            the shape of the input images (including channel-dimension,
            excluding batch-dimension)

        """
        super().__init__()
        self._img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(reduce(mul, self._img_shape))),
            torch.nn.Tanh()
        )

    def forward(self, z):
        """
        Forwards a noise batch through the generator

        Parameters
        ----------
        z : :class:`torch.Tensor`
            the noise batch

        Returns
        -------
        :class:`torch.Tensor`
            the generated image batch

        """
        img = self.model(z)
        img = img.view(img.shape[0], *self._img_shape)
        return img


class Discriminator(torch.nn.Module):
    """
    A simple Discriminator Network
    """

    def __init__(self, img_shape):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the input images (including channel-dimension,
            excluding batch-dimension)

        """
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(reduce(mul, img_shape)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        """
        Forwards a single (generated or real) image batch through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the classification result of the discriminator

        """
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
