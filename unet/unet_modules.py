import torch
import torch.nn as nn
import torch.nn.functional as f


class DoubleConvolution(nn.Module):
    """
    Class used to initialize the conv 3x3, ReLu step.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        mid_channels : int
            Number if mid-layer channels
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Preforms and returns results from the conv 3x3, ReLu step.

        Parameters
        ----------
        x : torch.tensor
            Input data
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Class used to initialize the max pool 2x2 step.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Preforms and returns results from the max pool 2x2 step.

        Parameters
        ----------
        x : torch.tensor
            Input data
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Class used to initialize the up-conv 2x2 step.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        bilinear : bool
            Bilinear interpolation in upsampling(default)
        """

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvolution(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Preforms and returns results from the up-conv 2x2 step.

        Parameters
        ----------
        x1 : torch.tensor
            From
        x2 : torch.tensor
            To
        """
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = f.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvolution(nn.Module):
    """
    Class used to initialize the conv 1x1 step.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        """
        super(OutConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Preforms and returns results from the conv 1x1 step.

        Parameters
        ----------
        x : torch.tensor
            Input data
        """
        return self.conv(x)
