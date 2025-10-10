import torch
import torch.nn as nn

#basic double conv block
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

# Simple 3D-UNet
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        # Encoder
        self.enc1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = double_conv(256, 512)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottom = double_conv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = double_conv(1024, 512)

        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)

        # Output layer
        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottom(self.pool4(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)