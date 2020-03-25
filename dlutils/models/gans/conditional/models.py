from functools import reduce
from operator import mul

import torch


class Generator(torch.nn.Module):
    """
    A very simple generator model to generate images of specific classes
    """

    def __init__(self, n_classes, img_shape, latent_dim):
        """

        Parameters
        ----------
        n_classes : int
            the total number of classes
        img_shape : tuple
            the shape of the input images (including channel-dimension,
            excluding batch-dimension)
        latent_dim : int
            the size of the latent dimension

        """
        super().__init__()

        self.label_emb = torch.nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(reduce(mul, img_shape))),
            torch.nn.Tanh()
        )

        self._img_shape = img_shape

    def forward(self, noise, labels):
        """
        Forwards a single noise batch with the according labels through the
        generator network

        Parameters
        ----------
        noise : :class:`torch.Tensor`
            the noise batch
        labels : :class:`torch.Tensor`
            the label batch

        Returns
        -------
        :class:`torch.Tensor`
            the generated image

        """
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels).squeeze(1), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self._img_shape)
        return img


class Discriminator(torch.nn.Module):
    """
    A very simple discriminator network for conditionally generated images
    """

    def __init__(self, n_classes, img_shape):
        """

        Parameters
        ----------
        n_classes : int
            the total number of classes
        img_shape : tuple
            the shape of the input images (including channel-dimension,
            excluding batch-dimension)

        """
        super().__init__()

        self.label_embedding = torch.nn.Embedding(n_classes, n_classes)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_classes + int(reduce(mul, img_shape)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        """
        Feeds an image and label batch through the discriminator network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch
        labels : :class:`torch.Tensor`
            the label batch

        Returns
        -------
        :class:`torch.Tensor`
            the discriminative result

        """
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1),
                          self.label_embedding(labels).squeeze(1)), -1)
        validity = self.model(d_in)
        return validity
