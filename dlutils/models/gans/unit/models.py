import torch


class ResidualBlock(torch.nn.Module):
    """
    A simple Residual Block
    """

    def __init__(self, num_filts):
        """

        Parameters
        ----------
        num_filts : int
            number of filters to use inside this block
        """
        super().__init__()

        conv_block = [
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num_filts, num_filts, 3),
            torch.nn.InstanceNorm2d(num_filts),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num_filts, num_filts, 3),
            torch.nn.InstanceNorm2d(num_filts),
        ]

        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Applies the block's layers and the residual skip connection

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the residual block's output tensor

        """
        return x + self.conv_block(x)


class Encoder(torch.nn.Module):
    """
    A simple Encoder Network
    """

    def __init__(
            self,
            shared_block,
            in_channels=3,
            num_filts=64,
            n_downsample=2):
        """

        Parameters
        ----------
        shared_block :
            the shared block to compute the mean tensor
        in_channels : int
            the channels fo the input images
        num_filts : int
            number of filters to use
        n_downsample : int
            number of downsampling stages
        """
        super().__init__()

        # Initial convolution block
        layers = [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels, num_filts, 7),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                torch.nn.Conv2d(num_filts, num_filts * 2, 4, stride=2,
                                padding=1),
                torch.nn.InstanceNorm2d(num_filts * 2),
                torch.nn.ReLU(inplace=True),
            ]
            num_filts *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(num_filts)]

        self.model_blocks = torch.nn.Sequential(*layers)
        self.shared_block = shared_block

    @staticmethod
    def reparameterization(mu):
        """
        Reparametrizes the model with a normal distribution

        Parameters
        ----------
        mu : :class:`torch.Tensor`
            the mean value

        Returns
        -------
        :class:`torch.Tensor`
            the reparametrized tensor

        """
        return torch.randn_like(mu) + mu

    def forward(self, x):
        """
        Feeds a single batch through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch to encode

        Returns
        -------
        :class:`torch.Tensor`
            the mean value
        :class:`torch.Tensor`
            the reparametrized mean

        """
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return mu, z


class Generator(torch.nn.Module):
    """
    A residual generator network
    """

    def __init__(self, shared_block, out_channels=3, num_filts=64,
                 n_upsample=2):
        """

        Parameters
        ----------
        shared_block :
            the block, which is shared betweeen the generators
        out_channels : int
            number of channels contained in the output image
        num_filts : int
            number of filters to use
        n_upsample : int
            number of upsampling stages
        """
        super().__init__()

        self.shared_block = shared_block

        layers = []
        num_filts = num_filts * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(num_filts)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                torch.nn.ConvTranspose2d(num_filts, num_filts // 2, 4,
                                         stride=2, padding=1),
                torch.nn.InstanceNorm2d(num_filts // 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]
            num_filts = num_filts // 2

        # Output layer
        layers += [torch.nn.ReflectionPad2d(3),
                   torch.nn.Conv2d(num_filts, out_channels, 7),
                   torch.nn.Tanh()]

        self.model_blocks = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Feeds a single batch through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch to translate

        Returns
        -------
        :class:`torch.Tensor`
            the genearted images
        """
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x


class Discriminator(torch.nn.Module):
    """
    A discriminative PatchGAN-based network
    """

    def __init__(self, img_shape):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the input images (including channels,
            excluding batch dimension)

        """
        super().__init__()
        channels, height, width = img_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4,
                                      stride=2, padding=1)]
            if normalize:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            torch.nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, img):
        """
        Feeds a single batch through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the images' validity
        """
        return self.model(img)
