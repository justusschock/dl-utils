import torch

__all__ = [
    'ConvNd',
    'PoolingNd',
    'NormNd',
    'DropoutNd'
]


class ConvNd(torch.nn.Module):
    """
    Convolution Wrapper to Switch accross dimensions and transposed by a
    single argument
    See Also
    --------
    Torch Convolutions:
        * :class:`torch.nn.Conv1d`
        * :class:`torch.nn.Conv2d`
        * :class:`torch.nn.Conv3d`
        * :class:`torch.nn.ConvTranspose1d`
        * :class:`torch.nn.ConvTranspose2d`
        * :class:`torch.nn.ConvTranspose3d`
    """

    def __init__(self, n_dim, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 transposed=False, **kwargs):
        """
        Parameters
        ----------
        n_dim : int
            number of dimensions the convolution should be chosen for
        in_channels : int
            input channels
        out_channels : int
            output_channels
        kernel_size : int or Iterable
            size of convolution kernel
        stride : int or Iterable
            convolution stride
        padding : int or Iterable
            padding value
            (if input or output padding depends on whether the convolution
            is transposed or not)
        dilation : int or Iterable
            convolution dilation
        groups : int
            number of convolution groups
        bias : bool
            whether to include bias or not
        transposed : bool
            whether the convolution should be transposed or not
        """
        super().__init__()

        if transposed:
            transposed_str = "Transpose"
        else:
            transposed_str = ""

        conv_cls = getattr(torch.nn, "Conv%s%dd" % (transposed_str, n_dim))

        self.conv = conv_cls(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the actual convolution's forward
        Parameters
        ----------
        x : :class:`torch.Tensor`
            input tensor
        Returns
        -------
        :class:`torch.Tensor`
            the convolved input
        """
        return self.conv(x)


class PoolingNd(torch.nn.Module):
    """
    Wrapper to switch between different pooling types and convolutions by a
    single argument
    See Also
    --------
    Torch Pooling Classes:
        * :class:`torch.nn.MaxPool1d`
        * :class:`torch.nn.MaxPool2d`
        * :class:`torch.nn.MaxPool3d`
        * :class:`torch.nn.AvgPool1d`
        * :class:`torch.nn.AvgPool2d`
        * :class:`torch.nn.AvgPool3d`
        * :class:`torch.nn.AdaptiveMaxPool1d`
        * :class:`torch.nn.AdaptiveMaxPool2d`
        * :class:`torch.nn.AdaptiveMaxPool3d`
        * :class:`torch.nn.AdaptiveAvgPool1d`
        * :class:`torch.nn.AdaptiveAvgPool2d`
        * :class:`torch.nn.AdaptiveAvgPool3d`
    """

    def __init__(self, pooling_type, n_dim, *args, **kwargs):
        """
        Parameters
        ----------
        pooling_type : str
            Type of Pooling, case sensitive.
                Supported values are
                * ``Max``
                * ``Avg``
                * ``AdaptiveAvg``
                * ``AdaptiveMax``
        n_dim : int
            number of dimensions
        *args :
            positional arguments of the chosen pooling class
        **kwargs :
            keyword arguments of the chosen pooling class
        """
        super().__init__()

        pool_cls = getattr(torch.nn, "%sPool%dd" % (pooling_type, n_dim))

        self.pool = pool_cls(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the actual pooling's forward
        Parameters
        ----------
        x : :class:`torch.Tensor`
            input tensor
        Returns
        -------
        :class:`torch.Tensor`
            the pooled input
        """
        return self.pool(x)


class NormNd(torch.nn.Module):
    """
    Wrapper to switch between different types of normalization and
    dimensions by a single argument
    See Also
    --------
    Torch Normalizations:
            * :class:`torch.nn.BatchNorm1d`
            * :class:`torch.nn.BatchNorm2d`
            * :class:`torch.nn.BatchNorm3d`
            * :class:`torch.nn.InstanceNorm1d`
            * :class:`torch.nn.InstanceNorm2d`
            * :class:`torch.nn.InstanceNorm3d`
            * :class:`torch.nn.LocalResponseNorm`
            * :class:`torch.nn.GroupNorm`
            * :class:`torch.nn.LayerNorm`
    """

    def __init__(self, norm_type, n_dim, *args, **kwargs):
        """
        Parameters
        ----------
        norm_type : str
            type of normalization, case sensitive.
            Supported types are:
                * ``Batch``
                * ``Instance``
                * ``LocalResponse``
                * ``Group``
                * ``Layer``
        n_dim : int or None
            dimension of normalization input; can be None if normalization
            is dimension-agnostic (e.g. LayerNorm)
        *args :
            Positional arguments of chosen normalization class
        **kwargs :
            keyword arguments of chosen normalization class
        """
        super().__init__()
        if norm_type is None:
            self.norm = None
        else:
            if n_dim is None:
                dim_str = ""
            else:
                dim_str = str(n_dim)

            norm_cls = getattr(torch.nn, "%sNorm%sd" % (norm_type, dim_str))
            self.norm = norm_cls(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the actual normalization's forward
        Parameters
        ----------
        x : :class:`torch.Tensor`
            input tensor
        Returns
        -------
        :class:`torch.Tensor`
            the normalized input
        """
        if self.norm is None:
            return x
        else:
            return self.norm(x)


class DropoutNd(torch.nn.Module):
    """
        Wrapper to switch between different types of normalization and
        dimensions by a single argument
        See Also
        --------
        Torch Dropouts:
                * :class:`torch.nn.Dropout1d`
                * :class:`torch.nn.Dropout2d`
                * :class:`torch.nn.Dropout3d`
        """
    def __init__(self, n_dim, p=0.5, inplace=False):
        """

        Parameters
        ----------
        n_dim : int
            dimension of dropout input
        p : float
            dropout rate
        inplace : bool
            whether to apply the dropout inplace
        """
        super().__init__()
        dropout_cls = getattr(torch.nn, "Dropout%dd" % n_dim)
        self.dropout = dropout_cls(p=p, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
       Calls the actual dropout's forward
       Parameters
       ----------
       x : :class:`torch.Tensor`
           input tensor
       Returns
       -------
       :class:`torch.Tensor`
           the dropout output
       """
        return self.dropout(x)