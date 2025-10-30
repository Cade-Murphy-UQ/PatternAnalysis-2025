import torch
import torch.nn as nn
import torch.nn.functional as F
"""
3D U-Net–style segmentation network inspired by the Improved UNet paper's context/localization pathway with deep supervision.
"""

# 3x3x3 convolution
def convolution(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )

class ContextBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.3):
        super().__init__()
        self.core = nn.Sequential(
            convolution(in_ch, out_ch),
            nn.Dropout3d(p_drop),
            convolution(out_ch, out_ch),
        )
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        y = self.core(x)
        return F.leaky_relu(y + self.proj(x), negative_slope=0.01, inplace=True)

# 3x3x3 stride 2 convolution
def stride2_convolution(ch):
    return convolution(ch, ch, k=3, s=2, p=1)

# upsampling module
def upsampling_module(in_ch, out_ch):
    return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)

# localization module
def localization_module(in_ch, mid_ch, out_ch):
    return nn.Sequential(
        convolution(in_ch, mid_ch, k=1, s=1, p=0),
        convolution(mid_ch, out_ch, k=3, s=1, p=1),
    )

# segmentation layer
def segmentation_layer(in_ch, out_classes):
    return nn.Conv3d(in_ch, out_classes, kernel_size=1, bias=True)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, p_drop=0.3):
        super().__init__()
        # encoder context and stride2
        self.enc1 = ContextBlock(in_channels, 16, p_drop)
        self.down1 = stride2_convolution(16)

        self.enc2 = ContextBlock(16, 32, p_drop)
        self.down2 = stride2_convolution(32)

        self.enc3 = ContextBlock(32, 64, p_drop)
        self.down3 = stride2_convolution(64)

        self.enc4 = ContextBlock(64, 128, p_drop)
        self.down4 = stride2_convolution(128)

        # bottleneck
        self.bottleneck = ContextBlock(128, 256, p_drop)

        # decoder, upsample then concatenate then localization
        self.up4  = upsampling_module(256, 128)
        self.loc4 = localization_module(128 + 128, 128, 128)

        self.up3  = upsampling_module(128, 64)
        self.loc3 = localization_module(64 + 64, 64, 64)

        self.up2  = upsampling_module(64, 32)
        self.loc2 = localization_module(32 + 32, 32, 32)

        self.up1  = upsampling_module(32, 16)
        self.loc1 = localization_module(16 + 16, 16, 16)

        # deep supervision heads for later element wise sum
        self.head_main = segmentation_layer(16, out_channels)
        self.head_d2   = segmentation_layer(32, out_channels)
        self.head_d3   = segmentation_layer(64, out_channels)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x);  d1 = self.down1(e1)
        e2 = self.enc2(d1); d2 = self.down2(e2)
        e3 = self.enc3(d2); d3 = self.down3(e3)
        e4 = self.enc4(d3); d4 = self.down4(e4)

        # bottleneck
        b = self.bottleneck(d4)

        # decoder, upsample then concatenate then localization
        u4 = self.up4(b)
        x4 = self.loc4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(x4)
        x3 = self.loc3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(x3)
        x2 = self.loc2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(x2)
        x1 = self.loc1(torch.cat([u1, e1], dim=1))

        # deep supervision 
        # upsample aux logits to the main head's spatial size
        logits = self.head_main(x1)
        target_size = logits.shape[2:]

        aux2 = F.interpolate(self.head_d2(x2), size=target_size, mode='nearest')
        aux3 = F.interpolate(self.head_d3(x3), size=target_size, mode='nearest')

        #element wise sum calculation
        return logits + 0.4 * aux2 + 0.2 * aux3