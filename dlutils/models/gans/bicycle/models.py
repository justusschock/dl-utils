import torch
from torchvision.models import resnet18


class UNetDown(torch.nn.Module):
    """
    A Downsampling block to build a generic unet
    """

    def __init__(self, in_size, out_size, normalize=True):
        """
        Parameters
        ----------
        in_size : int
            the number of input channels
        out_size : int
            the number of output channels
        normalize : bool
            whether to apply normalization
        """
        super(UNetDown, self).__init__()
        layers = [torch.nn.Conv2d(in_size, out_size, 3, stride=2, padding=1,
                                  bias=False)]
        if normalize:
            layers.append(torch.nn.BatchNorm2d(out_size, 0.8))
        layers.append(torch.nn.LeakyReLU(0.2))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Feeds a single tensor through the whole block
        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input image batch
        Returns
        -------
        :class:`torch.Tensor`
            The result tensor
        """
        return self.model(x)


class UNetUp(torch.nn.Module):
    """
    A basic upsampling block to build generic unets
    """

    def __init__(self, in_size, out_size):
        """
        Parameters
        ----------
        in_size : int
            number of input channels
        out_size : int
            number of output channels
        """
        super(UNetUp, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_size, out_size, 3, stride=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(out_size, 0.8),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        """
        Feeds a single tensor through the whole block
        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input image batch
        Returns
        -------
        :class:`torch.Tensor`
            The result tensor
        """
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(torch.nn.Module):
    """
    Class to build a generic U-Net like generator
    """

    def __init__(self, latent_dim, img_shape):
        """
        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension
        img_shape : tuple
            the shape of the image batches (including channel-dimension,
            excluding batch-dimension)
        """
        super().__init__()
        channels, self.h, self.w = img_shape

        self.fc = torch.nn.Linear(latent_dim, self.h * self.w)

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, channels, 3, stride=1, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x, z):
        """
        Forwards a image batch and a noise vector through the network
        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        z : :class:`torch.Tensor`
            the noise batch
        Returns
        -------
        :class:`torch.Tensor`
            the result batch
        """
        # Propogate noise through fc layer and reshape to img shape
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)
        d1 = self.down1(torch.cat((x, z), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


class Encoder(torch.nn.Module):
    """
    A very simple encoder network (basically a resnet18)
    """

    def __init__(self, latent_dim):
        """
        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension
        """
        super().__init__()
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = torch.nn.Sequential(
            *list(resnet18_model.children())[:-3])
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = torch.nn.Linear(256, latent_dim)
        self.fc_logvar = torch.nn.Linear(256, latent_dim)

    def forward(self, img):
        """
        Feeds an image batch through the network to encode it
        Parameters
        ----------
        img : :class:`torch.Tensor`
            an image batch
        Returns
        -------
        :class:`torch.Tensor`
            the encoded image batch
        """
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


class MultiResolutionDiscriminator(torch.nn.Module):
    """
    A discriminator class working on multiple resolutions
    """

    def __init__(self, input_channels):
        """
        Parameters
        ----------
        input_channels : int
            the number of input channels
        """
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                      padding=1)]
            if normalize:
                layers.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2))
            return layers

        # Extracts discriminator models
        self.models = torch.nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                torch.nn.Sequential(
                    *discriminator_block(input_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    torch.nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = torch.nn.AdaptiveAvgPool2d(1, stride=2,
                                                     padding=[1, 1],
                                                     count_include_pad=False)

    def forward(self, x):
        """
        Forwards the image batch through all resolution and downsamples it
        afterwards to prepare the input for the next resolution
        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        Returns
        -------
        list of :class:`torch.Tensor`
            the classification results for each resolution
        """
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
