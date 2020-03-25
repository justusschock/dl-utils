import torch


class Generator(torch.nn.Module):
    """
    A Basic generator network
    """

    def __init__(self, n_channels, latent_dim, img_size):
        """

        Parameters
        ----------
        n_channels : int
            the number of image channels
        latent_dim : int
            the size of the latent dimension
        img_size : int
            the size of the squared images (per side)

        """
        super().__init__()

        self.init_size = img_size // 4
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
            torch.nn.Conv2d(64, n_channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, noise):
        """
        Forwards a noise batch through the network

        Parameters
        ----------
        noise : :class:`torch.Tensor`
            the noise vector to generate images from

        Returns
        -------
        :class:`torch.Tensor`
            the generated image batch

        """
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    """
    A simple discriminator network
    """

    def __init__(self, n_channels, img_size):
        """

        Parameters
        ----------
        n_channels : int
            the number of image channels
        img_size : int
            the size of the squared images (per side)
        """
        super().__init__()

        # Upsampling
        self.down = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 64, 3, 2, 1),
            torch.nn.ReLU())
        # Fully-connected layers
        self.down_size = img_size // 2
        down_dim = 64 * (img_size // 2) ** 2
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(down_dim, 32),
            torch.nn.BatchNorm1d(32, 0.8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, down_dim),
            torch.nn.BatchNorm1d(down_dim),
            torch.nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, n_channels, 3, 1, 1))

    def forward(self, img):
        """
        Forwards a single image batch through the discriminator

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the discriminator's prediction

        """
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(
            out.view(
                out.size(0),
                64,
                self.down_size,
                self.down_size))
        return out
