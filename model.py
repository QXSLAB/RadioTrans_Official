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
            
            nn.Conv2d(9, ch//2, 4, 4, 12),
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


class TransLayer(nn.Module):

    def __init__(self, H, ch, L, dim, head, layer):

        super().__init__()

        stride = H//L
        self.to_emb = nn.Linear(stride**2*ch, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, L**2, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, head),
                                                 layer, nn.LayerNorm(dim))
        self.from_emb = nn.Linear(dim, stride**2*ch)

        self.stride = stride
        self.L = L

    def forward(self,x):

        skip = x
        s, L = self.stride, self.L
        
        # shape [batch, ch, H, H]
        x = einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=s, p2=s)
        # shape [batch, L**2, ch*stride**2]
        x = self.to_emb(x)
        # shape [batch, L**2, dim]
        x += self.pos_emb*0
        # shape [batch, L**2, dim]
        x = self.transformer(x)
        # shape [batch, L**2, dim]
        x = self.from_emb(x)
        # shape [batch, L**2, ch*stride**2]
        x = einops.rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', h=L, p1=s, p2=s)
        # shape [batch, ch, H, H]

        x = x+skip

        return x


class Unet(nn.Module):


    def __init__(self, ch):

        super().__init__()
        
        self.land = LandFeature(ch)

        self.down1 = DoubleC(ch, 2*ch)  # batch x ch*2 x 64 x 64
        self.down2 = Down(2*ch, 4*ch)  # batch x ch*4 x 32 x 32
        self.down3 = Down(4*ch, 8*ch)  # batch x ch*8 x 16 x 16
        self.down4 = Down(8*ch, 16*ch)  # batch x ch*16 x 8 x 8

        self.up3 = Up(ch*16, ch*8)  # batch, ch*8, 16, 16
        self.up2 = Up(ch*8, ch*4)  # batch, ch*4, 16, 16
        self.up1 = Up(ch*4, ch*2)  # batch, ch*2, 16, 16

        self.trans1 = TransLayer(64, ch*2, 8, 512, 8, 1)
        self.trans2 = TransLayer(32, ch*4, 8, 512, 8, 1)
        self.trans3 = TransLayer(16, ch*8, 8, 512, 8, 1)
        self.trans4 = TransLayer(8, ch*16, 8, 512, 8, 1)
        
        self.outConv = nn.Sequential(
            DoubleC(ch*2, ch),
            nn.Conv2d(ch, 1, 1, 1, 0),
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
        inp = torch.cat([x, grid, xyz], dim=1)

        inp = self.land(inp)

        x1 = self.down1(inp)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up3(self.trans4(x4), x3)
        x = self.up2(self.trans3(x), x2)
        x = self.up1(self.trans2(x), x1)
        x = self.trans1(x)

        x = self.outConv(x)

        return x
