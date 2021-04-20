"""
    models for radio propagation effect (RPE) generation
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
import einops


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


class LandFeature(nn.Module):
    """
        extract feature of landscape
        input batch x 2 x 1000 x 1000
        output batch x ch x 64 x 64
    """

    def __init__(self, ch):

        super().__init__()

        self.main = nn.Sequential(
            
            nn.Conv2d(4, ch//2, 4, 4, 12),
            nn.BatchNorm2d(ch//2),
            nn.ReLU(),

            nn.Conv2d(ch//2, ch, 4, 4, 0),
            nn.BatchNorm2d(ch),
            nn.ReLU())

    def forward(self, x):

        return self.main(x)


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


    def __init__(self, ch):

        super().__init__()

        self.land = LandFeature(ch)

        self.inc = DoubleC(ch, ch*2)  # batch x ch*2 x 64 x 64
        self.down1 = Down(ch*2, ch*4)  # batch x ch*4 x 32 x 32
        self.down2 = Down(ch*4, ch*8)  # batch x ch*8 x 16 x 16
        self.down3 = Down(ch*8, ch*16)  # batch x ch*16 x 8 x 8

        self.up3 = Up(ch*16, ch*8)  # batch x ch*8 x 16 x 16
        self.up2 = Up(ch*8, ch*4)  # batch x ch*4 x 32 x 32
        self.up1 = Up(ch*4, ch*2)  # batch x ch*2 x 64 x 64

        self.outConv = nn.Sequential(
            nn.Conv2d(ch*2, 1, 1, 1, 0),
            nn.Sigmoid())

    def forward(self, inp):

        inp = self.land(inp)

        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.outConv(x)


class Decode(nn.Module):
    """
        Up scale then double conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2, 0)
        self.conv = DoubleC(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Trans(nn.Module):

    def __init__(self, dim, head=8, layer=4):

        super().__init__()

        self.dim, self.head, self.layer = dim, head, layer
        self.L = 16*16

        self.pad = nn.ZeroPad2d(12)

        self.patch_emb = nn.Sequential(
            nn.Conv2d(9, dim//4, 4, 4, 0), # shape 256
            nn.BatchNorm2d(dim//4),
            nn.ReLU(),
            nn.Conv2d(dim//4, dim//2, 4, 4, 0), # shape 64
            nn.BatchNorm2d(dim//2),
            nn.ReLU(),
            nn.Conv2d(dim//2, dim, 4, 4, 0), # shape 16
            nn.BatchNorm2d(dim),
            nn.ReLU())

        self.pos_emb = nn.Parameter(torch.randn(1, self.L, self.dim))
        
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.dim, self.head),
                                                 self.layer, nn.LayerNorm(self.dim))
        self.refine = nn.Sequential(
            Decode(dim*2, dim), # 32
            Decode(dim, dim//2), # 64
            nn.Conv2d(dim//2, 1, 3, 1, 1),
            nn.Sigmoid())

    def forward(self, x):

        b, c, h, w = x.shape

        # define grid for each pixel
        grid_x, grid_y = x.new_ones((b, 1, h, w)), x.new_ones((b, 1, h, w))
        range_x = einops.repeat(torch.arange(h)/h, "h -> b 1 h w", b=b, w=w)
        range_x = range_x.to(x.device)
        range_y = einops.repeat(torch.arange(w)/w, "w -> b 1 h w", b=b, h=h)
        range_y = range_y.to(x.device)
        grid_x, grid_y = grid_x*range_x, grid_y*range_y
        grid = torch.cat([grid_x, grid_y], dim=1)

        # get xyz loc for source
        s_map = x[:,[2],:,:]
        z = s_map*(s_map>0)
        z = einops.reduce(z, "b c h w -> b c", "max")
        xy = grid*(s_map>0)
        xy = einops.reduce(xy, "b c h w -> b c", "max")
        xyz = torch.cat([xy, z], dim=1)
        xyz = einops.repeat(xyz, "b c -> b c h w", h=h, w=w)

        # combine
        x = torch.cat([x, grid, xyz], dim=1)

        # shape [batch, 5, 1000, 1000]
        x = self.pad(x)
        # shape [batch, 5, 1024, 1024]
        x = self.patch_emb(x)
        skip = x
        # shape [batch, dim, 16, 16]
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        # shape [batch, self.L, self.dim]
        x += self.pos_emb
        # shape [batch, self.L, self.dim]
        x = self.transformer(x)
        # shape [batch, self.L, self.dim]
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=16)
        # shape [batch, self.dim, 16, 16]
        x = torch.cat([x, skip], dim=1)
        # shape [batch, 2*self.dim, 16, 16]
        x = self.refine(x)
        # shape [batch, 1, 64, 64]

        return x

