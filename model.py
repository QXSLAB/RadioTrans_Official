"""
    models for radio propagation effect (RPE) generation
"""

import torch
import torch.nn as nn


def weight_init(m):

    """
        weight initialization
    """

    classname = m.__class__.__name__
    if not classname.find("Conv") == -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
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
            nn.LayerNorm([16, 16]),
            nn.LeakyReLU(0.2),
            # (batch, 256, 16, 16)
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LayerNorm([8, 8]),
            nn.LeakyReLU(0.2),
            # (batch, 512, 8, 8)
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.LayerNorm([4, 4]),
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
