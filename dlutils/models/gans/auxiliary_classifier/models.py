import torch


class Generator(torch.nn.Module):
    """
    A very simple generator model, which uses an embedding as conditioning
    for creating samples of certain classes
    """

    def __init__(self, n_classes=10, latent_dim=100, img_size=32,
                 n_channels=1):
        """

        Parameters
        ----------
        n_classes : int
            the number of classes
        latent_dim : int
            the size of the latent dimension
        img_size : int
            the size of the squared images (per dimension)
        n_channels : int
            the number of image channels

        """
        super().__init__()

        self.label_emb = torch.nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(
                latent_dim,
                128 * self.init_size ** 2))

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
            torch.nn.Conv2d(64, n_channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, noise, labels):
        """
        Forwards noise and labels through the generator

        Parameters
        ----------
        noise : :class:`torch.Tensor`
            the noise vector
        labels : :class:`torch.Tensor`
            the label vector

        Returns
        -------
        :class:`torch.Tensor`
            the generated images

        """
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    """
    A very simple discriminator model
    """

    def __init__(self, n_classes=10, img_size=32, n_channels=1):
        """

        Parameters
        ----------
        n_classes : int
            number of classes
        img_size : int
            the size of the squared images (per dimension)
        n_channels : int
            the number of image channels

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
            *discriminator_block(n_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.conv_blocks(torch.rand(1, n_channels, img_size,
                                              img_size)).size(2)

        # Output layers
        self.adv_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size ** 2, 1), torch.nn.Sigmoid())
        self.aux_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size ** 2, n_classes), torch.nn.Softmax())

    def forward(self, img):
        """
        Forwards the images through the discriminator and classificator network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the (generated or real) image batch

        Returns
        -------
        :class:`torch.Tensor`
            the validity of the image
        :class:`torch.Tensor`
            the predicted label
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
