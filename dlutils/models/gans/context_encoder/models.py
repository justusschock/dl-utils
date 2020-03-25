import torch


class Generator(torch.nn.Module):
    """
    Simple Generator Network
    """

    def __init__(self, in_channels=3):
        """

        Parameters
        ----------
        in_channels : int
            number of image channels
        """
        super().__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Conv2d(in_feat, out_feat, 4, stride=2,
                                      padding=1)]
            if normalize:
                layers.append(torch.nn.BatchNorm2d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [torch.nn.ConvTranspose2d(in_feat, out_feat, 4,
                                               stride=2, padding=1)]
            if normalize:
                layers.append(torch.nn.BatchNorm2d(out_feat, 0.8))
            layers.append(torch.nn.ReLU())
            return layers

        self.model = torch.nn.Sequential(
            *downsample(in_channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            torch.nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            torch.nn.Conv2d(64, in_channels, 3, 1, 1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        """
        Feeds a single tensor through the generative network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input tensor

        Returns
        -------
        :class:`torch.Tensor`
            the generative output

        """
        return self.model(x)


class Discriminator(torch.nn.Module):
    """
    Simple discriminator network
    """

    def __init__(self, in_channels=3):
        """

        Parameters
        ----------
        in_channels : int
            number of input channels
        """
        super().__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True),
                                               (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_channels, out_filters, stride,
                                              normalize))
            in_channels = out_filters

        layers.append(torch.nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, img):
        """
        Feeds an image batch through the discriminator network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the discriminative result

        """
        return self.model(img)
