import torch


class ResidualBlock(torch.nn.Module):
    """
    A single Residual Block
    """

    def __init__(self, num_filts):
        """

        Parameters
        ----------
        num_filts : int
            number of input and output channels/filters
        """
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(num_filts, num_filts, 3, 1, 1),
            torch.nn.BatchNorm2d(num_filts),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_filts, num_filts, 3, 1, 1),
            torch.nn.BatchNorm2d(num_filts),
        )

    def forward(self, x):
        """
        Computes the residual output for a single batch

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the image batch

        """
        return x + self.block(x)


class Generator(torch.nn.Module):
    """
    The generator topology
    """

    def __init__(self, latent_dim, num_channels, img_size, n_residual_blocks,
                 num_filts=64):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        num_channels : int
            number of image channels to generate
        img_size : int
            number of pixels per side of the image
        n_residual_blocks : int
            number of residual blocks inside the generator
        num_filts : int
            number of filters inside each of the blocks
        """
        super().__init__()

        # Fully-connected layer which constructs image channel shaped output
        # from noise
        self.fc = torch.nn.Linear(latent_dim, num_channels * img_size ** 2)

        self.l1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels * 2, 64, 3, 1, 1),
            torch.nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(n_residual_blocks):
            resblocks.append(ResidualBlock(num_filts))
        self.resblocks = torch.nn.Sequential(*resblocks)

        self.l2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, num_channels, 3, 1, 1), torch.nn.Tanh())

    def forward(self, img, z):
        """
        Feeds a set of batches through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image tensor
        z : :class:`torch.Tensor`
            the noise tensor

        Returns
        -------
        :class:`torch.Tensor`
            batch of generated images

        """
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class Discriminator(torch.nn.Module):
    """
    A discriminator network
    """

    def __init__(self, num_channels):
        """

        Parameters
        ----------
        num_channels : int
            number of channels per input image
        """
        super().__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [torch.nn.Conv2d(in_features, out_features, 3, stride=2,
                                      padding=1),
                      torch.nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_features))
            return layers

        self.model = torch.nn.Sequential(
            *block(num_channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            torch.nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        """
        Feeds a single batch through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            a single image batch

        Returns
        -------
        :class:`torch.Tensor`
            the resulting validity batch

        """
        validity = self.model(img)

        return validity


class Classifier(torch.nn.Module):
    """
    Classifier Network
    """

    def __init__(self, num_channels, img_size, n_classes):
        """

        Parameters
        ----------
        num_channels : int
            number of image channels
        img_size : int
            number of pixels per side
        n_classes : int
            number of classes
        """
        super().__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [torch.nn.Conv2d(in_features, out_features, 3,
                                      stride=2, padding=1),
                      torch.nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_features))
            return layers

        self.model = torch.nn.Sequential(
            *block(num_channels, 64, normalization=False), *block(64, 128),
            *block(128, 256), *block(256, 512)
        )

        # downsampled size
        dsize = self.model(torch.rand(1, num_channels, img_size, img_size)
                           ).size(2)
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512 * dsize ** 2, n_classes),
            torch.nn.Softmax())

    def forward(self, img):
        """
        Feeds a single batch through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the resulting label batch

        """
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label
