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


class CGAN_G(nn.Module):

    """
        generator of MLP cgan
        ref https://raw.githubusercontent.com/znxlwm/
        pytorch-MNIST-CelebA-cGAN-cDCGAN/master/pytorch_cGAN.png
    """

    def __init__(self):

        super(CGAN_G, self).__init__()

        self.fc1_1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())

        self.fc1_2 = nn.Sequential(
            nn.Linear(4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())

        self.main = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),

            nn.Linear(8192, 4096),
            nn.Tanh())

    def forward(self, noise, param):

        feat1 = self.fc1_1(noise)
        feat2 = self.fc1_2(param)
        feature = torch.cat([feat1, feat2], dim=1)
        img = self.main(feature)

        return img


class CGAN_D(nn.Module):

    """
        discriminator of MLP cgan
        ref https://raw.githubusercontent.com/znxlwm/
        pytorch-MNIST-CelebA-cGAN-cDCGAN/master/pytorch_cGAN.png
    """

    def __init__(self):

        super(CGAN_D, self).__init__()

        self.fc1_1 = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.LeakyReLU(0.2))

        self.fc1_2 = nn.Sequential(
            nn.Linear(4, 8192),
            nn.LeakyReLU(0.2))

        self.main = nn.Sequential(
            nn.Linear(16384, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid())


    def forward(self, param, img):

        img = img.view(-1, 64*64)
        feat1 = self.fc1_1(img)
        feat2 = self.fc1_2(param)
        feature = torch.cat([feat1, feat2], dim=1)
        flag = self.main(feature)
        flag = flag.view(-1)

        return flag


class C_DCGAN_G(nn.Module):

    """
        generator of DCGAN cgan
        ref https://github.com/znxlwm/pytorch-MNIST-
        CelebA-cGAN-cDCGAN/blob/master/pytorch_cDCGAN.png
    """

    def __init__(self):

        super(C_DCGAN_G, self).__init__()

        self.deconv1_1 = nn.Sequential(
            # (batch, 100, 1, 1)
            nn.ConvTranspose2d(100, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU())  # (batch, 256, 4, 4)

        self.deconv1_2 = nn.Sequential(
            # (batch, 4, 1, 1)
            nn.ConvTranspose2d(4, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU())  # (batch, 256, 4, 4)

        self.main = nn.Sequential(
            # (batch, 512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (batch, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (batch, 128, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (batch, 64, 32, 32)
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh())  # (batch, 1, 64, 64)

    def forward(self, noise, param):

        noise = noise.reshape(-1, 100, 1, 1)
        param = param.reshape(-1, 4, 1, 1)
        feat1 = self.deconv1_1(noise)
        feat2 = self.deconv1_2(param)
        feature = torch.cat([feat1, feat2], dim=1)
        img = self.main(feature)

        return img


class C_DCGAN_D(nn.Module):

    """
        discriminator for DCGAN cgan
        ref https://github.com/znxlwm/pytorch-MNIST-
        CelebA-cGAN-cDCGAN/blob/master/pytorch_cDCGAN.png
    """

    def __init__(self):

        super(C_DCGAN_D, self).__init__()

        self.conv1_1 = nn.Sequential(
            # (batch, 1, 64, 64)
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2))  # (batch, 64, 32, 32)

        self.conv1_2 = nn.Sequential(
            # (batch, 4, 64, 64)
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2))  # (batch, 64, 32, 32)

        self.main = nn.Sequential(
            # (batch, 128, 32, 32)
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            # (batch, 256, 16, 16)
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # (batch, 512, 8, 8)
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2),
            # (batch, 1024, 4, 4)
            nn.Conv2d(1024, 1, 4, 1, 0))  # (batch, 1, 1, 1)


    def forward(self, param, img):

        feat1 = self.conv1_1(img)
        param = param.reshape(-1, 4, 1, 1)
        param = param.expand(-1, -1, 64, 64)
        feat2 = self.conv1_2(param)
        feature = torch.cat([feat1, feat2], dim=1)
        flag = self.main(feature)
        flag = flag.view(-1)

        return flag


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


class C_ResNet_G(nn.Module):

    """
        generator of cgan-wgan_gp-resnet
        ref Improved Training of Wasserstein GANs.pdf (Appendix F)
    """ 

    def __init__(self):

        super(C_ResNet_G, self).__init__()

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

    def forward(self, noise, param):

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


class C_ResNet_D(nn.Module):

    """
        discriminator for cgan-wgan_gp-ResNet
        ref Improved Training of Wasserstein GANs.pdf (Appendix F)
    """

    class BasicBlock(nn.Module):

        """
            block used to build ResNet-like model
        """

        def __init__(self, down):

            super(C_ResNet_D.BasicBlock, self).__init__()

            # (batch, 128, 2*w, 2*w)
            self.proj = nn.Conv2d(128, 128, 3, 2, 1) if down else None
            # (batch, 128, w/2*w, w/2*w)

            self.main = nn.Sequential(
                # (batch, 128, 2*w, 2*w)
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
                # (batch, 128, 2*w, 2*w)
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 2 if down else 1, 1),
                # (batch, 128, w/2*w, w/2*w)
                )

        def forward(self, x):

            skip = self.proj(x) if self.proj else x
            out = self.main(x)

            return out+skip


    def __init__(self):

        super(C_ResNet_D, self).__init__()

        # (batch, 1, 64, 64)
        self.conv1_1 = nn.Conv2d(1, 64, 4, 2, 1)
        # (batch, 64, 32, 32)

        # (batch, 4, 64, 64)
        self.conv1_2 = nn.Conv2d(4, 64, 4, 2, 1)
        # (batch, 64, 32, 32)

        self.main = nn.Sequential(
            # (batch, 128, 32, 32)
            C_ResNet_D.BasicBlock(True),
            # (batch, 128, 16, 16)
            C_ResNet_D.BasicBlock(True),
            # (batch, 128, 8, 8)
            C_ResNet_D.BasicBlock(False),
            # (batch, 128, 8, 8)
            C_ResNet_D.BasicBlock(False),
            # (batch, 128, 8, 8)
            nn.ReLU(),
            nn.AvgPool2d(8),
            # (batch, 128, 1, 1)
            nn.Conv2d(128, 1, 1, 1, 0))
            # (batch, 1, 1, 1)

    def forward(self, param, img):

        feat1 = self.conv1_1(img)
        param = param.reshape(-1, 4, 1, 1)
        param = param.expand(-1, -1, 64, 64)
        feat2 = self.conv1_2(param)
        feature = torch.cat([feat1, feat2], dim=1)
        flag = self.main(feature)
        flag = flag.view(-1)

        return flag
