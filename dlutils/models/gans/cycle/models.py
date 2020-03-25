import torch


class ResidualBlock(torch.nn.Module):
    """
    A Simple Residual Block
    """

    def __init__(self, in_features):
        """

        Parameters
        ----------
        in_features : int
            number of input channels (will also be used for number of output
            channels)

        """
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        """
        Forwards a the inputs through the layer

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input batch

        Returns
        -------
        :class:`torch.Tensor`
            the output batch

        """
        return x + self.block(x)


class GeneratorResNet(torch.nn.Module):
    """
    A simple residual generator network
    """

    def __init__(self, input_channels, num_residual_blocks):
        """

        Parameters
        ----------
        input_channels : int
            number of channels (for input and output=
        num_residual_blocks : int
            number of residual blocks to use

        """
        super().__init__()

        # Initial convolution block
        out_features = 64
        model = [
            torch.nn.ReflectionPad2d(input_channels),
            torch.nn.Conv2d(input_channels, out_features, 7),
            torch.nn.InstanceNorm2d(out_features),
            torch.nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                torch.nn.Conv2d(in_features, out_features, 3, stride=2,
                                padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(in_features, out_features, 3, stride=1,
                                padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [torch.nn.ReflectionPad2d(input_channels),
                  torch.nn.Conv2d(out_features, input_channels, 7),
                  torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        """
        Forwards the input batch through the whole generator network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the input batch

        Returns
        -------
        :class:`torch.Tensor`
            the modules output batch

        """
        return self.model(x)


class Discriminator(torch.nn.Module):
    """
    A patchwise simple discriminator (PatchGAN)
    """

    def __init__(self, input_shape):
        """

        Parameters
        ----------
        input_shape : tuple
            the shape of the input batch (including channels, excluding batch
            dimension)
        """
        super().__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                      padding=1)]
            if normalize:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        """
        Forwards the input batch through the whole generator network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the input batch

        Returns
        -------
        :class:`torch.Tensor`
            the modules output batch

        """
        return self.model(img)
