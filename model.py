import torch
import torch.nn as nn


def weight_init(m):
    
    classname = m.__class__.__name__
    if not classname.find("Conv") == -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif not classname.find("BatchNorm") == -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CGAN_G(nn.Module):
    
    """
        ref https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN
    """

    def __init__(self):

        super(CGAN_G, self).__init__()

        self.g_fc1_1 = nn.Sequential(                
                nn.Linear(100, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

        self.g_fc1_2 = nn.Sequential(
                nn.Linear(4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

        self.g = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 8192),
                nn.BatchNorm1d(8192),
                nn.ReLU(),

                nn.Linear(8192, 4096),
                nn.Tanh()
            )
    
    
    def forward(self, z, x):
        
        z = self.g_fc1_1(z)
        x = self.g_fc1_2(x)
        c = torch.cat([z, x], dim=1)
        y = self.g(c)

        return y


class CGAN_D(nn.Module):
    
    """
        ref https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN
    """

    def __init__(self):

        super(CGAN_D, self).__init__()

        self.d_fc1_1 = nn.Sequential(
                nn.Linear(4096, 8192),
                nn.LeakyReLU(0.2)
            )

        self.d_fc1_2 = nn.Sequential(
                nn.Linear(4, 8192),
                nn.LeakyReLU(0.2),
            )

        self.d = nn.Sequential(
                nn.Linear(16384, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                
                nn.Linear(256, 1),
                nn.Sigmoid()
            )


    def forward(self, x, y):

        y = y.view(-1, 64*64)

        x = self.d_fc1_2(x)
        y = self.d_fc1_1(y)
        c = torch.cat([x,y], dim=1)
        o = self.d(c)

        o = o.view(-1)

        return o
