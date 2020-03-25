import torch


class Encoder(torch.nn.Module):
    """
    A simple Encoder network, which encodes the content and the style
    separately
    """

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2,
                 style_dim=8):
        """

        Parameters
        ----------
        in_channels : int
            number of channels per input image
        dim : int
            number of filters
        n_residual : int
            number of residual blocks
        n_downsample : int
            number of downsampling blocks
        style_dim : int
            size of the style dimension
        """
        super().__init__()
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual,
                                              n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample,
                                          style_dim)

    def forward(self, x):
        """
        Encodes an image batch by forwarding it through the style and the
        content encoder

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the encoded image content
        :class:`torch.Tensor`
            the encoded image style

        """
        content_code = self.content_encoder(x)
        style_code = self.style_encoder(x)
        return content_code, style_code


class Decoder(torch.nn.Module):
    """
    Simple Decoder to convert a style encoding and a content encoding into an
    image
    """

    def __init__(self, out_channels=3, num_filts=64, n_residual=3,
                 n_upsample=2, style_dim=8):
        """

        Parameters
        ----------
        out_channels : int
            number of image channels to generate
        num_filts : int
            number if filters
        n_residual : int
            number of residual blocks
        n_upsample : int
            number of upsampling blocks
        style_dim : int
            size of the style encoding
        """
        super().__init__()

        layers = []
        num_filts = num_filts * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(num_filts, norm="adain")]

        # Upsampling
        for _ in range(n_upsample):
            layers += [torch.nn.Upsample(scale_factor=2),
                       torch.nn.Conv2d(num_filts,
                                       num_filts // 2,
                                       5,
                                       stride=1,
                                       padding=2),
                       torch.nn.LayerNorm(num_filts // 2),
                       torch.nn.ReLU(inplace=True),
                       ]
            num_filts = num_filts // 2

        # Output layer
        layers += [torch.nn.ReflectionPad2d(3),
                   torch.nn.Conv2d(num_filts, out_channels, 7),
                   torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*layers)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """
        Return the number of AdaIN parameters needed by the model

        Returns
        -------
        int
            number of AdaIN parameters
        """
        num_adain_params = 0
        for m in self.modules():
            if isinstance(m, AdaptiveInstanceNorm2d):
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """
        Assigns the adain_params to the AdaIN layers in model

        Parameters
        ----------
        adain_params : :class:`torch.Tensor`
            the AdaIN parameters, which should be assigned to the
            ada_in parameters
        """
        for m in self.modules():
            if isinstance(m, AdaptiveInstanceNorm2d):
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features: 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward(self, content_code, style_code):
        """
        Generates an image batch using the style and content encodings

        Parameters
        ----------
        content_code : :class:`torch.Tensor`
            content encoding
        style_code : :class:`torch.Tensor`
            style encoding

        Returns
        -------
        :class:`torch.Tensor`
            the generated image batch

        """
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img


class ContentEncoder(torch.nn.Module):
    """
    Encoder Network to encode an image's content
    """

    def __init__(
            self,
            in_channels=3,
            num_filts=64,
            n_residual=3,
            n_downsample=2):
        """

        Parameters
        ----------
        in_channels : int
            number of channels per input image
        num_filts : int
            number of filters per layer to use
        n_residual : int
            number of residual blocks
        n_downsample : int
            number of downsampling blocks

        """
        super().__init__()

        # Initial convolution block
        layers = [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels, num_filts, 7),
            torch.nn.InstanceNorm2d(num_filts),
            torch.nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [torch.nn.Conv2d(num_filts,
                                       num_filts * 2,
                                       4,
                                       stride=2,
                                       padding=1),
                       torch.nn.InstanceNorm2d(num_filts * 2),
                       torch.nn.ReLU(inplace=True),
                       ]
            num_filts *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(num_filts, norm="in")]

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Encodes the content of a single image batch by feeding it through the
        network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the content encoding

        """
        return self.model(x)


class StyleEncoder(torch.nn.Module):
    """
    Encoder Network to encode an image's style
    """

    def __init__(self, in_channels=3, num_filts=64, n_downsample=2,
                 style_dim=8):
        """

        Parameters
        ----------
        in_channels : int
            number of input channels per image
        num_filts : int
            number of filters per layer
        n_downsample : int
            number of downsampling blocks
        style_dim : int
            size of the style encoding dimension
        """
        super().__init__()

        # Initial conv block
        layers = [torch.nn.ReflectionPad2d(3),
                  torch.nn.Conv2d(in_channels, num_filts, 7),
                  torch.nn.ReLU(inplace=True)]

        # Downsampling
        for _ in range(2):
            layers += [torch.nn.Conv2d(num_filts, num_filts * 2, 4, stride=2,
                                       padding=1),
                       torch.nn.ReLU(inplace=True)]
            num_filts *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):
            layers += [torch.nn.Conv2d(num_filts, num_filts, 4, stride=2,
                                       padding=1),
                       torch.nn.ReLU(inplace=True)]

        # Average pool and output layer
        layers += [torch.nn.AdaptiveAvgPool2d(1),
                   torch.nn.Conv2d(num_filts, style_dim, 1, 1, 0)]

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Encodes the style of a single image batch by feeding it through the
        network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the style encoding

        """
        return self.model(x)


class MLP(torch.nn.Module):
    """
    A simple Multilayer Perceptron to transform the AdaIN parameters
    """

    def __init__(self, input_dim, output_dim, num_filts=256, n_blk=3):
        """

        Parameters
        ----------
        input_dim : int
            number of features per input sample
        output_dim : int
            number of features per output sample
        num_filts : int
            number of filters per layer
        n_blk : int
            number of layer blocks
        """
        super().__init__()
        layers = [torch.nn.Linear(input_dim, num_filts),
                  torch.nn.ReLU(inplace=True)]

        for _ in range(n_blk - 2):
            layers += [torch.nn.Linear(num_filts, num_filts),
                       torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(num_filts, output_dim)]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Transforms the style encoding of a single image batch by feeding it
        through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the transformed style encoding

        """
        return self.model(x.view(x.size(0), -1))


class MultiResolutionDiscriminator(torch.nn.Module):
    """
    A discriminator operating on multiple Paths each with a different
    resolution
    """

    def __init__(self, in_channels=3, num_paths=3):
        """

        Parameters
        ----------
        in_channels : int
            number of channels per input image
        num_paths : int
            number of parallel discriminator paths (each with lower resolution
            than the ones before)
        """
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                      padding=1)]
            if normalize:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = torch.nn.ModuleList()
        for i in range(num_paths):
            self.models.add_module(
                "disc_%d" % i,
                torch.nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    torch.nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = torch.nn.AvgPool2d(in_channels, stride=2,
                                             padding=[1, 1],
                                             count_include_pad=False)

    def forward(self, x):
        """
        Computes the outputs coming from the differen resolution paths

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        list of :class:`torch.Tensor`
            a list containing the outputs for the different resolutions

        """
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class ResidualBlock(torch.nn.Module):
    """
    Residual Block supporting the :class:`AdaptiveInstanceNorm2d`
    """

    def __init__(self, features, norm="in"):
        """

        Parameters
        ----------
        features : int
            number of filters to use for input and output
        norm : str
            must be one of 'adain' | 'in'

        Raises
        ------
        ValueError
            if :param:`norm` is not in ['adain', 'in']
        TypeError
            if :param:`norm` is not a string

        """
        super().__init__()

        if isinstance(norm, str):
            if norm == "adain":
                norm_layer = AdaptiveInstanceNorm2d
            elif norm == 'in':
                norm_layer = torch.nn.InstanceNorm2d
            else:
                raise ValueError("Invalid Normtype given. Got %s, but "
                                 "expected one of 'adain' | 'in'" % norm)

        else:
            raise TypeError("Invalid Normtype given. Got type %s but expected "
                            "str" % type(norm).__name__)

        self.block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(features, features, 3),
            norm_layer(features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        """
        Forwards an image batch through the entire residual block

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the result obtained by the residual block

        """
        return x + self.block(x)


class AdaptiveInstanceNorm2d(torch.nn.Module):
    """
    Adaptive Instance Normalization

    References
    ----------
    https://github.com/NVlabs/MUNIT/blob/master/networks.py
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """

        Parameters
        ----------
        num_features : int
            number of channels
        eps : float
            relative tolerance
        momentum : float
            momentum term for update

        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        """
        Forwards an image batch through the layer

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the normalized image batch

        Raises
        ------
        AssertionError
            if ``weight`` is None or ``bias`` is None

        """
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = torch.nn.functional.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"
