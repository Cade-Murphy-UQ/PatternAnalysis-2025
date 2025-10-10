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
