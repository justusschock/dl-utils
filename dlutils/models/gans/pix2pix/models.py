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

    def __init__(self, in_channels, out_channels):
        """
        Parameters
        ----------
        in_channels : int
            the number of channels for input images
        out_channels : int
            the number of channels for output images
        """
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(128, out_channels, 4, padding=1),
            torch.nn.Tanh(),
        )

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
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(torch.nn.Module):
    """
    Discriminator Network to decide whether a combination of image and
    condition image is real or fake
    """

    def __init__(self, src_channels, target_channels):
        """

        Parameters
        ----------
        src_channels : int
            number of image channels for images in the source domain (domain A)
        target_channels : int
            number of image channels for images in the target domain (domain B)
        """
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                      padding=1)]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(src_channels + target_channels,
                                 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_b, img_a):
        """
        Forwards images of both domains through discriminative network

        Parameters
        ----------
        img_b : :class:`torch.Tensor`
            the image of domain B (the target domain)
        img_a : :class:`torch.Tensor`
            the image of domain A (the source domain; as conditioning)

        Returns
        -------
        :class:`torch.Tensor`
            the discriminative result

        """
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_b, img_a), 1)
        return self.model(img_input)
