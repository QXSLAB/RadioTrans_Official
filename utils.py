from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import matplotlib.patches as patches 
from raw import PowerSet
import os
from math import floor
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from torch import autograd
import torch.nn as nn
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from raw import PowerSet
from model import Unet, weight_init
from main import setup_seed

def visualize_param(img, param, tail=None):

    img = img.detach().cpu().numpy()
    param = param.detach().cpu().numpy()

    f, x, y, a = param[0], param[1], param[2], param[3]

    x = ((x*0.5+0.5)*63).round()
    y = ((y*0.5+0.5)*63).round()

    l = 8
    a = a*3.14
    dx = l*np.cos(a)
    dy = l*np.sin(a)

    fig, ax = plt.subplots(1)

    ax.imshow(img)
    arrow = patches.Arrow(x, y, dx, dy, color='r', width=3, linewidth=1)
    circle = patches.Circle((x, y), radius=2, color='r')
    ax.add_patch(arrow)
    ax.add_patch(circle)

    plt.savefig("{}-{}.png".format(param, tail))

if __name__ == '__main__':

    setup_seed(99)

    # load data
    dset = PowerSet("/home/dell/hdd/space_effect_png",
                    transforms.Compose([
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
    train_l = floor(0.8*len(dset))
    tset, vset = random_split(dset, [train_l, len(dset)-train_l])
    tloader = DataLoader(tset, batch_size=64,
                         shuffle=True, num_workers=4,
                         drop_last=True)
    vloader = DataLoader(vset, batch_size=64,
                         shuffle=False, num_workers=4)

    g = Unet().cuda()
    g.load_state_dict(torch.load("/home/dell/hdd/program_fsrpe/unet_l1_loss/G_best.pt"))

    g.eval()
    all_match, all_param, all_fake = [], [], []
    for param, match in tqdm(vloader):
        param, match = param.cuda(), match.cuda()
        fake = g(param)
        mae = l1_loss(fake, match, reduction="none")
        mae = mae.sum(dim=[1,2,3])/(64*64)
        mask = mae>0.15
        all_match.append(match[mask].detach().cpu())
        all_param.append(param[mask].detach().cpu())
        all_fake.append(fake[mask].detach().cpu())
    all_match = torch.cat(all_match, dim=0)
    all_param = torch.cat(all_param, dim=0)
    all_fake = torch.cat(all_fake, dim=0)
    all_diff = all_match-all_fake

    for i in tqdm(range(len(all_match))):
        visualize_param(all_diff[i, 0], all_param[i], "diff")
        #visualize_param(all_match[i, 0], all_param[i], "match")
        #visualize_param(all_fake[i, 0], all_param[i], "fake")

