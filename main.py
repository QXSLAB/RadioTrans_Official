"""
    main file
"""

import os
from math import floor
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from raw import PowerSet
from model import CGAN_G, CGAN_D, weight_init
from model import C_DCGAN_G, C_DCGAN_D


def quality(img, match, metric):

    """
        evaluate generated image
    """

    return metric(img.view(-1), match.view(-1))


def display_quality(grid, metric, qlty):

    """
        display quality in image grid
    """

    grid_np = grid.detach().cpu().numpy()
    grid_np = np.uint8(grid_np.transpose(1, 2, 0)*255)
    grid_pil = Image.fromarray(grid_np)
    draw = ImageDraw.Draw(grid_pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/"
                              "dejavu/DejaVuSansMono.ttf", size=30)
    draw.text((0, 0), "{0} {1:.5f}".format(metric.__name__, qlty),
              fill=(0, 255, 0), font=font)

    return grid_pil


def visualize_weight_grad(writer, model, step):

    """
        visualize weight and grad in tensorboard
    """

    cls = model.__class__.__name__
    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        tag = "{}/{}".format(cls, tag)
        writer.add_histogram("weight/"+tag, value, step)
        writer.add_histogram("grad/"+tag, value.grad, step)


def visualize_gen(G, fixed_batch, metric, msg, writer=None):

    """
        visualize and qualify generated image
    """

    # generate fake image
    param, match = fixed_batch
    noise = torch.rand(param.shape[0], 100).cuda()
    fake = G(noise, param).reshape(-1, 1, 64, 64)
    fake_grid = make_grid(fake[:64], normalize=True)

    # compare with match image
    match_grid = make_grid(match[:64], normalize=True)
    diff_grid = torch.abs(fake_grid - match_grid)

    # quality annotation
    qlty = quality(fake, match, metric)
    fake_pil = display_quality(fake_grid, metric, qlty)
    diff_pil = display_quality(diff_grid, metric, qlty)
    match_pil = display_quality(match_grid, metric, qlty)

    if not writer:
        fake_pil.save(os.path.join(msg, "best.png"))
        diff_pil.save(os.path.join(msg, "diff.png"))
        match_pil.save(os.path.join(msg, "match.png"))
    else:
        fake_np = np.asarray(fake_pil).transpose(2, 0, 1)
        writer.add_image("fake/{}".format(msg), fake_np)

        diff_np = np.asarray(diff_pil).transpose(2, 0, 1)
        writer.add_image("diff/{}".format(msg), diff_np)


def main():

    trail = "dcgan_imp_func"

    experiment = "/home/dell/hdd/program_fsrpe/{}".format(trail)

    # make dir to save result
    if os.path.exists(experiment):
        yes = input("folder {}, overwrite? [Y/N]:".format(experiment))
        if yes.lower() == "y":
            os.system("trash {}".format(experiment))
        else:
            exit("make new folder")
    os.mkdir(experiment)

    # setup tensorboard
    writer = SummaryWriter(experiment)

    # load data
    dset = PowerSet("/home/dell/hdd/space_effect_png",
                    transforms.Compose([
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
    train_l = floor(0.8*len(dset))
    tset, vset = random_split(dset, [train_l, len(dset)-train_l])
    tloader = DataLoader(tset, batch_size=128,
                         shuffle=True, num_workers=4,
                         drop_last=True)
    # vloader = DataLoader(vset, batch_size=128,
    #                      shuffle=False, num_workers=4)

    # check data
    fixed, match_batch = next(iter(tloader))
    fixed, match_batch = fixed.cuda(), match_batch.cuda()
    match_grid = make_grid(match_batch[:64], normalize=True)
    writer.add_image("match", match_grid)

    # setup loss
    criterion = nn.BCELoss()

    # setup model
    # G = CGAN_G().cuda()
    # D = CGAN_D().cuda()
    G = C_DCGAN_G().cuda()
    D = C_DCGAN_D().cuda()

    # difine visualization model
    class CGAN(nn.Module):
        """
            for model visualization
        """
        def __init__(self, G, D):
            super(CGAN, self).__init__()
            self.G = G
            self.D = D
        def forward(self, param):
            noise = torch.rand(param.shape[0], 100).cuda()
            img = self.G(noise, param)
            flag = self.D(param, img)
            return flag

    # visulize model
    cgan = CGAN(G, D)
    writer.add_graph(cgan, fixed)

    # apply weight init
    G.apply(weight_init)
    D.apply(weight_init)

    # set optimizer
    G_opt = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    D_opt = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # record global step
    step = 0

    # recored best result
    best = float("inf")

    # setup metric
    metric = l1_loss

    for e in range(10000):

        iter_loader = iter(tloader)

        # run until exhausted
        while True:

            # grab data for D

            param, match = next(iter_loader, ("", ""))
            if match == "":
                break
            param, match = param.cuda(), match.cuda()

            _, no_match = next(iter_loader, ("", ""))
            if no_match == "":
                break
            no_match = no_match.cuda()

            noise = torch.rand(param.shape[0], 100).cuda()
            fake = G(noise, param).detach()

            # clear grad in D
            D.zero_grad()

            # grad of match
            flag = D(param, match)
            errD_match = criterion(flag, flag.new_ones(flag.shape))
            errD_match.backward()

            # grad of no_match
            flag = D(param, no_match)
            errD_no_match = criterion(flag, flag.new_zeros(flag.shape))
            errD_no_match.backward()

            # grad of fake
            flag = D(param, fake)
            errD_fake = criterion(flag, flag.new_zeros(flag.shape))
            errD_fake.backward()

            # update D
            D_opt.step()

            # grab data for G
            param, match = next(iter_loader, ("", ""))
            if param == "":
                break
            noise = torch.rand(param.shape[0], 100)
            noise = noise.cuda()
            param = param.cuda()
            match = match.cuda()

            # clear grad in G
            G.zero_grad()

            # grad of G
            img = G(noise, param)
            flag = D(param, img)
            errG = criterion(flag, flag.new_ones(flag.shape))
            errG.backward()

            # generation quality
            qlty = quality(img, match, metric)

            # update G
            G_opt.step()

            # update global step
            step += 1

            if step % 10 == 9:

                # track progress
                print("Epoch [{0:5d}] Global [{1:8d}] "
                      "errD [{2:2.5f}/{3:2.5f}/{4:2.5f}] "
                      "errG [{5:2.5f}] {6} [{7:2.5f}]".format(e, step,
                          errD_match, errD_no_match, errD_fake,
                          errG, metric.__name__, qlty))

                # track performance
                errD = {"match": errD_match,
                        "no_match": errD_no_match,
                        "fake": errD_fake,
                        "all": errD_match + errD_no_match + errD_fake}
                writer.add_scalars("err/D", errD, step)
                writer.add_scalar("err/G", errG, step)
                writer.add_scalar("err/{}".format(metric.__name__), qlty, step)

            if qlty < best:

                best = qlty

                # visualize best image
                visualize_gen(G, (fixed, match_batch), metric, experiment)

                # save best model
                torch.save(G.state_dict(),
                           os.path.join(experiment, "G_best.pt"))
                torch.save(D.state_dict(),
                           os.path.join(experiment, "D_best.pt"))

        if e % 10 == 9:

            # track weight and grad
            visualize_weight_grad(writer, G, step)
            visualize_weight_grad(writer, D, step)

            # track image generation
            visualize_gen(G, (fixed, match_batch), metric,
                          "{}".format(e), writer)


if __name__ == "__main__":
    main()
