import torch


class UNetDown(torch.nn.Module):
    """
    The Downsampling Block of a UNet
    """

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        """

        Parameters
        ----------
        in_size : int
            number of input channels
        out_size : int
            number of output channels
        normalize : bool
            whether to normalize or not
        dropout : float
            the dropout rate

        """
        super().__init__()
        model = [torch.nn.Conv2d(in_size, out_size, 4, stride=2, padding=1,
                                 bias=False)]
        if normalize:
            model.append(torch.nn.BatchNorm2d(out_size, 0.8))
        model.append(torch.nn.LeakyReLU(0.2))
        if dropout:
            model.append(torch.nn.Dropout(dropout))

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        """
        Forwards an input batch through the model

        Parameters
        ----------
        x : :class:`torch.Tensor`

        Returns
        -------
        :class:`torch.Tensor`
            the resulting batch

        """
        return self.model(x)


class UNetUp(torch.nn.Module):
    """
    An Upsampling Block of the UNet
    """

    def __init__(self, in_size, out_size, dropout=0.0):
        """

        Parameters
        ----------
        in_size : int
            number of input channels
        out_size : int
            number of output channels
        dropout : float
            dropout rate
        """
        super().__init__()
        model = [
            torch.nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1,
                                     bias=False),
            torch.nn.BatchNorm2d(out_size, 0.8),
            torch.nn.ReLU(inplace=True),
        ]
        if dropout:
            model.append(torch.nn.Dropout(dropout))

        self.model = torch.nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out


class Generator(torch.nn.Module):
    """
    A very simple Unet-like Generator network
    """

    def __init__(self, input_channels):
        """

        Parameters
        ----------
        input_channels : int
            Number of input channels

        """
        super().__init__()
        self.down1 = UNetDown(input_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128 + input_channels, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256 + input_channels, 64)

        final = [torch.nn.Upsample(scale_factor=2),
                 torch.nn.Conv2d(128, input_channels, 3, 1, 1),
                 torch.nn.Tanh()]
        self.final = torch.nn.Sequential(*final)

    def forward(self, x, x_lr):
        """
        Forwards a batch of images and low resolution images through the
        network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the masked image batch
        x_lr : :class:`torch.Tensor`
            the low resolution image batch

        Returns
        -------
        :class:`torch.Tensor`
            the resulting image batch

        """
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d2 = torch.cat((d2, x_lr), 1)
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
    The discriminator model

    """

    def __init__(self, input_shape):
        """

        Parameters
        ----------
        input_shape : tuple
            a tuple specifying the shape of input images (including channels,
            excluding batch dimension)

        """
        super().__init__()

        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        patch_h, patch_w = int(height / 2 ** 3), int(width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True),
                                               (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride,
                                              normalize))
            in_filters = out_filters

        layers.append(torch.nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, img):
        """
        Forwards an image batch through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`

        Returns
        -------
        :class:`torch.Tensor`

        """
        return self.model(img)
