import torch


class UNetDown(torch.nn.Module):
    """
    Downsampling Block for UNet Architecture
    """

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        """
        Parameters
        ----------
        in_size : int
            number if input channels
        out_size : int
            number of output channels
        normalize : bool
            whether to include batchnorm
        dropout : float
            dropout rate
        """
        super().__init__()
        layers = [torch.nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(torch.nn.InstanceNorm2d(out_size))
        layers.append(torch.nn.LeakyReLU(0.2))
        if dropout:
            layers.append(torch.nn.Dropout(dropout))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forwards a single batch
        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input batch
        Returns
        -------
        :class:`torch.Tensor`
            the output batch
        """
        return self.model(x)


class UNetUp(torch.nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [torch.nn.ConvTranspose2d(in_size, out_size, 4, 2, 1),
                  torch.nn.InstanceNorm2d(out_size),
                  torch.nn.ReLU(inplace=True)]
        if dropout:
            layers.append(torch.nn.Dropout(dropout))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x, skip_input):
        """
        Forwards a single batch
        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input batch coming from the lower resolution stage
        skip_input : :class:`torch.Tensor`
            the input batch coming from the same resolution stage
        Returns
        -------
        :class:`torch.Tensor`
            the output batch
        """
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(torch.nn.Module):
    """
    A Unet-like Generator model
    """

    def __init__(self, num_channels):
        """
        Parameters
        ----------
        num_channels : int
            the number of channels (for input and output)
        """
        super().__init__()
        self.down1 = UNetDown(num_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128,
                num_channels,
                4,
                stride=2,
                padding=1),
            torch.nn.Tanh())

    def forward(self, x):
        """
        Forwards a single batch
        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input batch
        Returns
        -------
        :class:`torch.Tensor`
            the output batch
        """
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


class Discriminator(torch.nn.Module):
    """
    A simple discriminative network working on patches (PatchGAN)
    """

    def __init__(self, input_shape):
        """
        Parameters
        ----------
        input_shape : tuple
            the shape of the input batches (including channels,
            excluding batch dimension)
        """
        super().__init__()

        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4,
                                      stride=2, padding=1)]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, img):
        """
        Forwards a single batch
        Parameters
        ----------
        img : :class:`torch.Tensor`
            the input batch
        Returns
        -------
        :class:`torch.Tensor`
            the output batch
        """
        # Concatenate image and condition image by channels to produce input
        return self.model(img)
