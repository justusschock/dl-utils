import torch


class ResidualBlock(torch.nn.Module):
    """
    Simple Residual Block
    """

    def __init__(self, num_filts):
        """

        Parameters
        ----------
        num_filts : int
            the number of filters/channels to use in this block
        """
        super().__init__()

        conv_block = [
            torch.nn.Conv2d(num_filts, num_filts, 3, stride=1, padding=1,
                            bias=False),
            torch.nn.InstanceNorm2d(num_filts, affine=True,
                                    track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_filts, num_filts, 3, stride=1, padding=1,
                            bias=False),
            torch.nn.InstanceNorm2d(num_filts, affine=True,
                                    track_running_stats=True),
        ]

        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Computes the block's residual output

        Parameters
        ----------
        x : :class:`torch.Tensor`

        Returns
        -------
        :class:`torch.Tensor`
            the residual output

        """
        return x + self.conv_block(x)


class GeneratorResNet(torch.nn.Module):
    """
    The Residual Generator
    """

    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the generated images, should contain the channel
            dimension, but not the batch dimension
        res_blocks : int
            number of residual blocks
        c_dim : int
            size of the code dimension

        """
        super().__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            torch.nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3,
                            bias=False),
            torch.nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                torch.nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1,
                                bias=False),
                torch.nn.InstanceNorm2d(curr_dim * 2, affine=True,
                                        track_running_stats=True),
                torch.nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                torch.nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    4,
                    stride=2,
                    padding=1,
                    bias=False),
                torch.nn.InstanceNorm2d(
                    curr_dim // 2,
                    affine=True,
                    track_running_stats=True),
                torch.nn.ReLU(
                    inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [torch.nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3),
                  torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x, c):
        """
        Forwards an image batch and a batch of encoded attributes through the
        network

        Parameters
        ----------
        x : :class:´torch.Tensor`
            the image batch
        c : :class:`torch.Tensor`
            the encoded attributes

        Returns
        -------
        :class:`torch.Tensor`
            the resulting images

        """
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)


class Discriminator(torch.nn.Module):
    """
    A Discriminative model with an auxiliary classification task
    """

    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the input images (should include the channels and
            exclude the batch dimension)
        c_dim : int
            the size of the code dimension for encoded attributes
        n_strided : int
            number of strided convolutions
        """
        super().__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                torch.nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                padding=1),
                torch.nn.LeakyReLU(0.01)
            ]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = torch.nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = torch.nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction; Automatically infer last kernel size to
        # result in an 1x1 image after last layer
        kernel_size = self.model(torch.rand(1, *img_shape)).size(2)
        self.out2 = torch.nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        """
        Feeds an image batch through the network to obtain the validity and
        classification result

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the validity output
        :class:`torch.Tensor`
            the classification result

        """
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)
