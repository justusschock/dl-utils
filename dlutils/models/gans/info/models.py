import torch


class Generator(torch.nn.Module):
    """
    Simple Generator Network
    """

    def __init__(
            self,
            latent_dim,
            n_classes,
            code_dim,
            img_size,
            num_channels):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        n_classes : int
            number of classes
        code_dim : int
            size of the code dimension
        img_size : int
            number of pixels per image side
        num_channels : int
            number of channels to generate

        """
        super().__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = torch.nn.Linear(input_dim, 128 * self.init_size ** 2)

        self.conv_blocks = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        """
        Forwards a single batch through the network

        Parameters
        ----------
        noise : :class:`torch.Tensor`
            the noise vector
        labels : :class:`torch.Tensor`
            the label batch
        code : :class:`torch.Tensor`
            the code

        Returns
        -------
        :class:`torch.Tensor`
            the image batch

        """
        gen_input = torch.cat((noise, labels.to(noise.dtype), code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    """
    A simple discriminator network
    """

    def __init__(self, code_dim, n_classes, num_channels, img_size):
        """

        Parameters
        ----------
        code_dim : int
            size of the code dimension
        n_classes : int
            number of image classes
        num_channels : int
            number of image channels
        img_size : int
            number of pixels per side
        """
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     torch.nn.LeakyReLU(0.2, inplace=True),
                     torch.nn.Dropout2d(0.25)]
            if bn:
                block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = torch.nn.Sequential(
            *discriminator_block(num_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.conv_blocks(torch.rand(1, num_channels, img_size,
                                              img_size)).size(2)

        # Output layers
        self.adv_layer = torch.nn.Linear(128 * ds_size ** 2, 1)
        self.aux_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size ** 2, n_classes), torch.nn.Softmax())
        self.latent_layer = torch.nn.Linear(128 * ds_size ** 2, code_dim)

    def forward(self, img):
        """
        Feeds a single image batch through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the validity for each image
        :class:`torch.Tensor`
            the predicted label for each image
        :class:`torch.Tensor`
            the predicted latent code for each image

        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)
        return validity, label, latent_code
