import torch
from torchvision.models import vgg19


class FeatureExtractor(torch.nn.Module):
    """
    A simple VGG19 Feature Extractor
    """

    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(
            *list(vgg19_model.features.children())[:18])

    def forward(self, img):
        """
        Extracts features of a given image batch

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch to extract features from

        Returns
        -------
        :class:`torch.Tensor`
            the extracted features

        """
        return self.feature_extractor(img)


class ResidualBlock(torch.nn.Module):
    """
    A simple residual block
    """

    def __init__(self, num_filts):
        """

        Parameters
        ----------
        num_filts : int
            number of filters to use inside this block
        """
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(num_filts, num_filts, kernel_size=3, stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(num_filts, 0.8),
            torch.nn.PReLU(),
            torch.nn.Conv2d(num_filts, num_filts, kernel_size=3, stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(num_filts, 0.8),
        )

    def forward(self, x):
        """
        Feeds the input through the network and applies the residual skip
        connection

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input tensor

        Returns
        -------
        :class:`torch.Tensor`
            the output after applying the residual skip connection onto the
            block's output

        """
        return x + self.conv_block(x)


class GeneratorResNet(torch.nn.Module):
    """
    Residual Generator Network
    """

    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        """

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_residual_blocks : int
            number of residual blocks inside this generator

        """
        super().__init__()

        # First layer
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=9, stride=1,
                            padding=4),
            torch.nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = torch.nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(64, 256, 3, 1, 1),
                torch.nn.BatchNorm2d(256),
                torch.nn.PixelShuffle(upscale_factor=2),
                torch.nn.PReLU(),
            ]
        self.upsampling = torch.nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, out_channels, kernel_size=9, stride=1,
                            padding=4),
            torch.nn.Tanh())

    def forward(self, x):
        """
        Feeds an input batch through the whole network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input tensor

        Returns
        -------
        :class:`torch.Tensor`
            the output tensor obtained by feeding the input through the network

        """
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(torch.nn.Module):
    """
    A discriminator network originally introduced in PatchGAN
    """

    def __init__(self, img_shape):
        """

        Parameters
        ----------
        img_shape : tuple
            the input shape (including channels, excluding batch dimension)

        """
        super().__init__()

        in_channels, in_height, in_width = img_shape

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = [torch.nn.Conv2d(in_filters, out_filters,
                                      kernel_size=3, stride=1, padding=1)]
            if not first_block:
                layers.append(torch.nn.BatchNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            layers.append(torch.nn.Conv2d(out_filters, out_filters,
                                          kernel_size=3, stride=2, padding=1))
            layers.append(torch.nn.BatchNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters,
                                              first_block=(i == 0)))
            in_filters = out_filters

        layers.append(torch.nn.Conv2d(out_filters, 1, kernel_size=3,
                                      stride=1, padding=1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, img):
        """
        Feeds an image batch through the entire discriminator

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the validity of the given images

        """
        return self.model(img)
