import torch


class CoupledGenerators(torch.nn.Module):
    """
    Network of coupled generators (coupling is done via parameter sharing)
    """

    def __init__(self, img_size, latent_dim, in_channels):
        """

        Parameters
        ----------
        img_size : int
            size of input images
        latent_dim : int
            size of latent dimension
        in_channels : int
            number of image channels

        """
        super().__init__()

        self.init_size = img_size // 4
        self.fc = torch.nn.Sequential(torch.nn.Linear(latent_dim,
                                                      128 * self.init_size ** 2))

        self.shared_conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
        )
        self.G1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, in_channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )
        self.G2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, in_channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, noise):
        """
        Feeds input noise through the generative networks

        Parameters
        ----------
        noise : :class:`torch.Tensor`
            the input noise

        Returns
        -------
        :class:`torch.Tensor`
            images of domain A
        :class:`torch.Tensor`
            images of domain B

        """
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(torch.nn.Module):
    """
    Coupled Discriminator Network to check the validity of 2
    (possibly generated) images in different domains. The coupling is done
    by parameter sharing
    """

    def __init__(self, in_channels, img_size):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            block.extend([torch.nn.LeakyReLU(0.2, inplace=True),
                          torch.nn.Dropout2d(0.25)])
            return block

        self.conv1 = torch.nn.Sequential(*discriminator_block(in_channels,
                                                              16, bn=False),
                                         *discriminator_block(16, 32))
        self.conv2 = torch.nn.Sequential(*discriminator_block(in_channels,
                                                              16, bn=False),
                                         *discriminator_block(16, 32))

        self.shared_conv = torch.nn.Sequential(
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.shared_clf_head = torch.nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        """
        Forwards the input batches through the generator network

        Parameters
        ----------
        img1 : :class:`torch.Tensor`
            the input image batch of domain A
        img2 : :class:`torch.Tensor`
            the input image batch of domain B

        Returns
        -------
        :class:`torch.Tensor`
            the validity of the images in domain A
        :class:`torch.Tensor`
            the validity of the images in domain B

        """
        intermediate_repr_1 = self.conv1(img1)
        intermediate_repr_2 = self.conv2(img2)

        features_1 = self.shared_conv(intermediate_repr_1)
        features_2 = self.shared_conv(intermediate_repr_2)

        validity_1 = self.shared_clf_head(features_1.size(0), -1)
        validity_2 = self.shared_clf_head(features_2.size(0), -1)

        return validity_1, validity_2
