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
