"""
    models for radio propagation effect (RPE) generation
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np


def weight_init(m):

    """
        weight initialization
    """

    classname = m.__class__.__name__
    if not classname.find("Conv") == -1:
        nn.init.xavier_normal_(m.weight.data)
    elif not classname.find("BatchNorm") == -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DoubleC(nn.Module):
    """
        (conv2d->BN->ReLU)*2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),)
    def forward(self, x):
        return self.main(x)


class Down(nn.Module):
    """
        Down scale then double conv
    """
    # TODO use maxpooling
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleC(in_ch, out_ch))
    def forward(self, x):
        return self.main(x)


class Up(nn.Module):
    """
        Up scale then double conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2, 0)
        self.conv = DoubleC(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Unet(nn.Module):

    """
        generator of cgan-wgan_gp-resnet
        ref Improved Training of Wasserstein GANs.pdf (Appendix F)
    """ 

    def __init__(self):

        super().__init__()

        self.geo = torch.from_numpy(np.load("mask.npy")).unsqueeze(0)
        self.geo = F.resize(self.geo, (64, 64)).reshape(1, 1, 64, 64)

        self.inc = DoubleC(1+4, 64)  # batch x 64 x 64 x 64
        self.down1 = Down(64, 128)  # batch x 128 x 32 x 32
        self.down2 = Down(128, 256)  # batch x 256 x 16 x 16
        self.down3 = Down(256, 512)  # batch x 512 x 8 x 8
        self.down4 = Down(512, 1024)  # batch x 1024 x 4 x 4

        self.up4 = Up(1024, 512)  # batch x 512 x 8 x 8
        self.up3 = Up(512, 256)  # batch x 256 x 16 x 16
        self.up2 = Up(256, 128)  # batch x 128 x 32 x 32
        self.up1 = Up(128, 64)  # batch x 64 x 64 x 64
        
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Tanh())

    def forward(self, param):

        bs, ch = param.shape
        param_map = param.new_ones((bs, ch, 64, 64))
        param = param.reshape(bs, ch, 1, 1).expand(-1, -1, 64, 64)
        param_map = param*param_map
        geo = self.geo.expand(bs, -1, -1, -1).to(param.device)
        x = torch.cat([param_map, geo], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.outConv(x)
