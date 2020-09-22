import torch.nn as nn
from torch.cuda.amp import autocast

from unet.unet_modules import DoubleConvolution, Down, Up, OutConvolution


class UNet(nn.Module):
    """
    Basic U-net structure as described in : O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks
    for biomedical image segmentation.” 2015.
    """

    def __init__(self, n_channels: int, n_classes: int, bilinear=True):
        """

        Parameters
        ----------
        n_channels : int
            Number of channels in input-data.
        n_classes : int
            Number of classes to segment.
        bilinear : bool
            Bilinear interpolation in upsampling(default)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.in_conv = DoubleConvolution(n_channels, 64)
        self.down_conv_1 = Down(64, 128)
        self.down_conv_2 = Down(128, 256)
        self.down_conv_3 = Down(256, 512)
        self.down_conv_4 = Down(512, 1024 // factor)

        self.up_conv_1 = Up(1024, 512 // factor, bilinear)
        self.up_conv_2 = Up(512, 256 // factor, bilinear)
        self.up_conv_3 = Up(256, 128 // factor, bilinear)
        self.up_conv_4 = Up(128, 64, bilinear)
        self.out_conv = OutConvolution(64, n_classes)

    @autocast()
    def forward(self, x):
        """
        Feed-forward function.

        Parameters
        ----------
        x : torch.tensor
            Input data
        """
        x1 = self.in_conv(x)
        x2 = self.down_conv_1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.down_conv_3(x3)
        x5 = self.down_conv_4(x4)

        x = self.up_conv_1(x5, x4)
        x = self.up_conv_2(x, x3)
        x = self.up_conv_3(x, x2)
        x = self.up_conv_4(x, x1)
        out = self.out_conv(x)

        return out
