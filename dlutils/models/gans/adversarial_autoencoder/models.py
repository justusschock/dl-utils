from functools import reduce
from operator import mul

import torch


class Encoder(torch.nn.Module):
    """
    A very simple generator model
    """

    def __init__(self, latent_dim, img_shape):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        img_shape : tuple
            the shape of the input image
        """
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(reduce(mul, img_shape, 1)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = torch.nn.Linear(512, latent_dim)
        self.logvar = torch.nn.Linear(512, latent_dim)

    @staticmethod
    def reparametrize(mu, logvar):
        """
        Reparametrizes the data from a normal distribution

        Parameters
        ----------
        mu : :class:`torch.Tensor`
            the mean vector to use for a multivariate normal distribution
        logvar : :class:`torch.Tensor`
            the logvars, to calculate the stds for the multivariate normal
            distribution from

        Returns
        -------
        :class:`torch.Tensor`
            the reparametrized vector

        """
        std = torch.exp(logvar / 2)
        return torch.normal(mu, std)

    def forward(self, img):
        """
        Feeds the image batch through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the encoded image batch

        """
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparametrize(mu, logvar)
        return z


class Decoder(torch.nn.Module):
    """
    Decodes an already encoded image signal
    """

    def __init__(self, latent_dim, img_shape):
        """

        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension
        img_shape : tuple
            the shape of the input image

        """
        super().__init__()
        self._img_shape = img_shape

        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, int(reduce(mul, img_shape, 1))),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        """
        Feeds an encoded signal through the network for decoding

        Parameters
        ----------
        z : :class:`torch.Tensor`
            the encoded signal


        Returns
        -------
        :class:`torch.Tensor`
            the decoded signal

        """
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self._img_shape)
        return img


class Generator(torch.nn.Module):
    """
    A Generator consisting of a combined encoder-decoder structure
    """

    def __init__(self, latent_dim, img_shape):
        """

        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension
        img_shape : tuple
            the image's shape
        """
        super().__init__()
        self.encoder = Encoder(latent_dim, img_shape)
        self.decoder = Decoder(latent_dim, img_shape)

    def forward(self, x):
        """
        Feeds a tensor through both, the encoder and the decoder

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The tensor to be encoded and whose encoded representation will be
            decoded again

        Returns
        -------
        dict
            a dictionary containing the encoded and decoded version of
            :param:`x`
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return {"encoded": encoded, "decoded": decoded}


class Discriminator(torch.nn.Module):
    """
    A Discriminator model, working on the latent dimension
    """

    def __init__(self, latent_dim):
        """

        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension

        """
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, z):
        """
        Feeds an encoded tensor through the discriminator network

        Parameters
        ----------
        z : :class:`torch.Tensor`
            the encoded tensor

        Returns
        -------
        :class:`torch.Tensor`
            the discriminator's output

        """
        validity = self.model(z)
        return validity
