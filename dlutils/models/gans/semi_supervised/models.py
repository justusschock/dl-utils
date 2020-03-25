import torch


class Generator(torch.nn.Module):
    """
    A Generative Network
    """

    def __init__(self, latent_dim, img_size, num_channels, num_classes=10):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        img_size : int
            number of pixels per side of the image
        num_channels : int
            number of image channels to generate
        num_classes : int
            number of classes

        """
        super().__init__()

        self.label_emb = torch.nn.Embedding(num_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = torch.nn.Linear(latent_dim, 128 * self.init_size ** 2)

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

    def forward(self, noise):
        """
        Generates an image batch by feeding a batch of noise vectors through
        the network

        Parameters
        ----------
        noise : :class:`torch.Tensor`
            the noise batch

        Returns
        -------
        :class:`torch.Tensor`
            the batch of generated images

        """
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    """
    A discriminator network to determine the validity of images
    """

    def __init__(self, img_size, num_channels, num_classes=10):
        """

        Parameters
        ----------
        img_size : int
            number of pixels per image side
        num_channels : int
            number of image channels
        num_classes : int
            number of classes

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
        self.adv_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size ** 2, 1), torch.nn.Sigmoid())
        self.aux_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size ** 2, num_classes + 1),
            torch.nn.Softmax())

    def forward(self, img):
        """
        Forwards an image batch to determine the validity and the class label

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the batch of images

        Returns
        -------
        :class:`torch.Tensor`
            the validity of the image batch (an image is valid if it wasn't
            generated)
        :class:`torch.Tensor`
            the class label the image belongs to

        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
