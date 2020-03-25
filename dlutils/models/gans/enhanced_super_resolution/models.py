import torch
from torchvision.models import vgg19


class FeatureExtractor(torch.nn.Module):
    """
    Basic Feature Extraction Network (VGG 19, only convolutional part)
    """

    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = torch.nn.Sequential(
            *list(vgg19_model.features.children())[:35])

    def forward(self, img):
        """
        Feeds an image batch through the feature extraction stage

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the extracted features

        """
        return self.vgg19_54(img)


class DenseResidualBlock(torch.nn.Module):
    """
    A Dense Residual block
    """

    def __init__(self, filters, res_scale=0.2):
        """

        Parameters
        ----------
        filters : int
            number of filters
        res_scale : float
            the residual scaling factor
        """
        super().__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [
                torch.nn.Conv2d(
                    in_features,
                    filters,
                    3,
                    1,
                    1,
                    bias=True)]
            if non_linearity:
                layers += [torch.nn.LeakyReLU()]
            return torch.nn.Sequential(*layers)

        self.blocks = torch.nn.ModuleList()

        for i in range(1, 6):
            self.blocks.append(block(in_features=i * filters,
                                     non_linearity=i != 5))

    def forward(self, x):
        """
        Feeds an image batch through the feature extraction stage

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the block's outputs

        """
        inputs = x
        out = None
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(torch.nn.Module):
    """
    Block adding a residual skip connection around a dense residual block
    """

    def __init__(self, filters, res_scale=0.2):
        """

        Parameters
        ----------
        filters : int
            number of filters
        res_scale : float
            the residual scaling factor
        """
        super().__init__()
        self.res_scale = res_scale
        self.dense_blocks = torch.nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )

    def forward(self, x):
        """
        Feeds an image batch through the feature extraction stage

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the block's outputs

        """
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(torch.nn.Module):
    """
    Generator Network with nested residuals around dense blocks
    """

    def __init__(self, num_channels, filters=64, num_res_blocks=16,
                 num_upsample=2):
        """

        Parameters
        ----------
        num_channels : int
            number of image channels
        filters : int
            number of filters
        num_res_blocks : int
            number of residual blocks
        num_upsample : int
            number of upsampling layers
        """
        super().__init__()

        # First layer
        self.conv1 = torch.nn.Conv2d(num_channels, filters, kernel_size=3,
                                     stride=1, padding=1)
        # Residual blocks
        self.res_blocks = torch.nn.Sequential(
            *[ResidualInResidualDenseBlock(filters)
              for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = torch.nn.Conv2d(filters, filters, kernel_size=3,
                                     stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                torch.nn.Conv2d(filters, filters * 4, kernel_size=3,
                                stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = torch.nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1,
                            padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(filters, num_channels, kernel_size=3, stride=1,
                            padding=1),
        )

    def forward(self, x):
        """
        Forwards a single image batch through the entire network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the low-resolution image batch

        Returns
        -------
        :class:`torch.Tensor`
            the resulting (high resolution) image batch

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
    The discriminative model
    """

    def __init__(self, input_shape):
        """

        Parameters
        ----------
        input_shape : tuple
            shape of input batches (including channels,
            excluding batch dimensions)
        """
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(torch.nn.Conv2d(in_filters, out_filters,
                                          kernel_size=3, stride=1, padding=1))
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
        Forwards a single image batch through the entire network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the resulting validity tensor

        """
        return self.model(img)
