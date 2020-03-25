from typing import Tuple

from dlutils.models.nd import ConvNd, NormNd, PoolingNd
import torch
from torch.nn import functional as F

__all__ = [
    'UNet'
]


class UNet(torch.nn.Module):
    """
    The :class:`UNetTorch` is a convolutional encoder-decoder neural
    network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Depending on the input argument, this implementation can also become a
    LinkNet
    Notes
    -----
    Differences to the original paper:
        * padding is used in 3x3 convolutions to prevent loss of border pixels
        * merging outputs does not require cropping due to (1)
        * residual connections can be used by specifying ``merge_mode='add'``
        * if non-parametric upsampling is used in the decoder pathway (
            specified by upmode='upsample'), then an additional 1x1 2d
            convolution occurs after upsampling to reduce channel
            dimensionality by a factor of 2. This channel halving happens
            with the convolution in the tranpose convolution (specified by
            ``upmode='transpose'``)
    References
    ----------
    https://arxiv.org/abs/1505.04597
    https://arxiv.org/abs/1707.03718
    """

    def __init__(self, num_classes, in_channels=1, depth=5,
                 start_filts=64, n_dim=2, norm_layer="Batch",
                 up_mode='transpose', merge_mode='concat',
                 final_activation='Softmax', padding_kwargs: dict = None):
        """
        Parameters
        ----------
        num_classes : int
            number of output classes
        in_channels : int
            number of channels for the input tensor (default: 1)
        depth : int
            number of MaxPools in the U-Net (default: 5)
        start_filts : int
            number of convolutional filters for the first conv (affects all
            other conv-filter numbers too; default: 64)
        up_mode : str
            type of upconvolution. Must be one of ['transpose', 'upsample']
            if 'transpose':
                Use transpose convolution for upsampling
            if 'upsample':
                Use bilinear Interpolation for upsampling (no additional
                trainable parameters)
            default: 'transpose'
        merge_mode : str
            mode of merging the two paths (with and without pooling). Must
            be one of ['concat', 'add']
            if 'concat':
                Concatenates along the channel dimension (Original UNet)
            if 'add':
                Adds both tensors (Residual behaviour; LinkNet)
            default: 'merge'
       final_activation: str
        padding_kwargs : dict
            arguments for padding. If not given defaults to the default
            arguments of torch.nn.functional.pad
        """
        super().__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        if padding_kwargs is None:
            padding_kwargs = {}

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self._norm_layer = norm_layer
        self.padding_kwargs = padding_kwargs

        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()

        self.conv_final = None

        self._build_model(
            n_dim=n_dim,
            num_classes=num_classes,
            in_channels=in_channels,
            depth=depth,
            start_filts=start_filts,
            norm_layer=norm_layer)

        self.reset_params()

        if (isinstance(final_activation, str)
                and final_activation.lower() == 'sigmoid'):
            self.final_activation = torch.nn.Sigmoid()
        elif (isinstance(final_activation, str)
              and final_activation.lower() == 'softmax'):
            self.final_activation = torch.nn.Softmax(dim=1)
        else:
            self.final_activation = None

    @staticmethod
    def weight_init(m):
        """
        Initializes weights with xavier_normal and bias with zeros
        Parameters
        ----------
        m : torch.nn.Module
            module to initialize
        """
        if isinstance(m, ConvNd):
            torch.nn.init.xavier_normal_(m.conv.weight)
            torch.nn.init.constant_(m.conv.bias, 0)

    def reset_params(self):
        """
        Initialize all parameters
        """
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x) -> dict:
        """
        Feed tensor through network
        Parameters
        ----------
        x : torch.Tensor
        Returns
        -------
        torch.Tensor
            Prediction
        """
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used during trainer. This means you need to use
        # torch.nn.CrossEntropyLoss is your trainer script,
        # as this module includes a softmax already.
        x = self.conv_final(x)

        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def _build_model(self, n_dim, num_classes, in_channels=1, depth=5,
                     start_filts=64, norm_layer='Batch') -> None:
        """
        Builds the actual model
        Parameters
        ----------
        num_classes : int
            number of output classes
        in_channels : int
            number of channels for the input tensor (default: 1)
        depth : int
            number of MaxPools in the U-Net (default: 5)
        start_filts : int
            number of convolutional filters for the first conv (affects all
            other conv-filter numbers too; default: 64)
        Notes
        -----
        The Helper functions and classes are defined within this function
        because ``delira`` offers a possibility to save the source code
        along the weights to completely recover the network without needing
        a manually created network instance and these helper functions have
        to be saved too.
        """

        outs = in_channels
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = _DownConv(n_dim, ins, outs, pooling=pooling,
                                  norm_layer=norm_layer)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = _UpConv(n_dim, ins, outs, up_mode=self.up_mode,
                              merge_mode=self.merge_mode,
                              padding_kwargs=self.padding_kwargs)
            self.up_convs.append(up_conv)

        self.conv_final = _conv1x1(n_dim, outs, num_classes)


def _conv3x3(n_dim: int, in_channels: int, out_channels: int,
             stride: Union[int, tuple, list] = 1,
             padding: Union[int, tuple, list] = 1,
             bias: bool = True, groups: int = 1) -> ConvNd:
    """
    functional interface to create appropriate convolutional layer

    Parameters
    ----------
    n_dim : int
        the dimensionality of the input
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    stride : int or tuple or list
        convolutional strides
    padding : int or tuple or list
        convolutional padding
    bias : bool
        whether to include a bias
    groups : int
        number of convolution groups

    Returns
    -------
    ConvNd
        assembled convolution

    """
    return ConvNd(
        n_dim,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def _upconv2x2(n_dim: int, in_channels: int, out_channels: int,
               mode: str = 'transpose') -> torch.nn.Module:
    """
    Creates a module to perform upsampling

    Parameters
    ----------
    n_dim : int
        the dimensionality of the input
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    mode : str
        mode for upsampling

    Returns
    -------
    torch.nn.Module
        the moule performing the upsampling (either on a convolutional basis
        or on an interpolation basis)

    """
    if mode == 'transpose':
        return ConvNd(
            n_dim,
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            transposed=True
        )
    else:
        # out_channels is always going to be the same
        # as in_channels
        if n_dim == 2:
            upsample_mode = "bilinear"
        elif n_dim == 3:
            upsample_mode = "trilinear"
        else:
            raise ValueError

        return torch.nn.Sequential(
            torch.nn.Upsample(mode=upsample_mode, scale_factor=2),
            _conv1x1(n_dim, in_channels, out_channels))


def _conv1x1(n_dim: int, in_channels: int, out_channels: int,
             groups: int = 1) -> ConvNd:
    """
    Creates a module containing the appropriate convolution

    Parameters
    ----------
    n_dim : int
        dimensionality of the input
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    groups : int
        number of convolution groups

    Returns
    -------
    ConvNd
        appropriate convolution

    """
    return ConvNd(
        n_dim,
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class _DownConv(torch.nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, n_dim: int, in_channels: int, out_channels: int,
                 pooling: bool = True, norm_layer: str = "Batch"):
        """

        Parameters
        ----------
        n_dim : int
            dimensionality of the input
        in_channels : int
            number of input channels
        out_channels : int
            number of output cannels
        pooling : bool
            whether to apply pooling or not
        norm_layer : str
            the kind of normalization layer to use
        """
        super().__init__()

        self.n_dim = n_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = _conv3x3(self.n_dim, self.in_channels,
                              self.out_channels)
        self.norm1 = NormNd(norm_layer, n_dim, self.out_channels)
        self.conv2 = _conv3x3(self.n_dim, self.out_channels,
                              self.out_channels)
        self.norm2 = NormNd(norm_layer, n_dim, self.out_channels)

        if self.pooling:
            self.pool = PoolingNd("Max", n_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Does the actual computation

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        torch.Tensor
            output before pooling

        """
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class _UpConv(torch.nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, n_dim: int, in_channels: int, out_channels: int,
                 merge_mode: str = 'concat', up_mode: str = 'transpose',
                 norm_layer: str = 'Batch',
                 padding_kwargs: dict = None):
        """

        Parameters
        ----------
        n_dim : int
            input dimensionality
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        merge_mode : str
            how to merge the inputs
        up_mode : str
            how to do the upsampling
        norm_layer : str
            what kind of norm to use
        padding_kwargs : dict
            additional padding kwargs
        """
        super().__init__()

        if padding_kwargs is None:
            padding_kwargs = {}

        self.padding_kwargs = padding_kwargs

        self.n_dim = n_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = _upconv2x2(self.n_dim, self.in_channels,
                                 self.out_channels,
                                 mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = _conv3x3(
                self.n_dim,
                2 * self.out_channels,
                self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = _conv3x3(self.n_dim,
                                  out_channels,
                                  self.out_channels)
        self.norm1 = NormNd(norm_layer, n_dim, self.out_channels)
        self.conv2 = _conv3x3(self.n_dim,
                              self.out_channels,
                              self.out_channels)
        self.norm2 = NormNd(norm_layer, n_dim, self.out_channels)

    def forward(self, from_down: torch.Tensor,
                from_up: torch.Tensor) -> torch.Tensor:
        """
        Does the actual computation.

        Performs up convolution, pads inputs in a way, they are corresponding
        to each other, merges them and then applies to convolutions followed
        by normalization and relu each

        Parameters
        ----------
        from_down : torch.Tensor
            input from lower resolution stage
        from_up : torch.Tensor
            input from skip connection

        Returns
        -------
        torch.Tensor
            output tensor (after all convolutions)

        """
        from_up = self.upconv(from_up)

        # calculate padding sizes (exclude batchsize and channels)
        diffs = (torch.tensor(from_down.shape[3::-1],
                              device=from_down.device)
                 - torch.tensor(from_up.shape[3::-1],
                                device=from_down.device))

        _pad_from_up = torch.clamp(diffs, 0)
        _pad_from_down = torch.clamp(-diffs, 0)
        pad_from_up, pad_from_down = [], []
        for tmp_up, tmp_down in zip(_pad_from_up, _pad_from_down):
            pad_from_up += [tmp_up // 2, tmp_up // 2 + tmp_up % 2]
            pad_from_down += [tmp_down // 2,
                              tmp_down // 2 + tmp_down % 2]

        from_up_padded = torch.nn.functional.pad(
            from_up, pad_from_up, **self.padding_kwargs)
        from_down_padded = torch.nn.functional.pad(
            from_down, pad_from_down, **self.padding_kwargs
        )

        if self.merge_mode == 'concat':
            x = torch.cat((from_up_padded, from_down_padded), 1)
        else:
            x = from_up_padded + from_down_padded
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x
