"""
    main file
"""

import os
from math import floor
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


def quality(img, match, metric):

    """
        evaluate generated image
    """

    return metric(img.view(-1), match.view(-1))


def visulize_weight_grad(writer, model, step):

    """
        visualize weight and grad in tensorboard
    """

    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        writer.add_histogram("weight/"+tag, value, step)
        writer.add_histogram("grad/"+tag, value.grad, step)


def main():

    trail = "format_code"

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
    fixed, real_batch = next(iter(tloader))
    fixed, real_batch = fixed.cuda(), real_batch.cuda()
    grid_img = make_grid(real_batch[:64], normalize=True)
    writer.add_image("real", grid_img)

    # setup loss
    criterion = nn.BCELoss()

    # setup model
    G = CGAN_G().cuda()
    D = CGAN_D().cuda()

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
            metric = quality(img, match, l1_loss)

            # update G

            G_opt.step()

            # update global step

            step += 1
            if step % 10 == 9:
                continue

            # track progress

            print("Epoch [{}] Global [{}]".format(e, step), end="\r")

            # track performance

            errD = {"real": errD_match,
                    "no_match": errD_no_match,
                    "gen": errD_fake,
                    "all": errD_match + errD_no_match + errD_fake}
            writer.add_scalars("err/D", errD, step)
            writer.add_scalar("err/G", errG, step)
            writer.add_scalar("err/mae", metric, step)

        # save model

        if e % 10 == 9:

            # check weight

            visulize_weight_grad(writer, G, step)
            visulize_weight_grad(writer, D, step)

            # check output

            noise = torch.rand(fixed.shape[0], 100).cuda()
            img = G(noise, fixed)
            metric = quality(img, real_batch, l1_loss)
            fake_img = make_grid(img[:64], normalize=True)
            writer.add_image("fake/{0}/{1}".format(e, metric), fake_img)

            torch.save(G.state_dict(),
                       os.path.join(experiment, "G_{}.pt".format(e)))
            torch.save(D.state_dict(),
                       os.path.join(experiment, "D_{}.pt".format(e)))


if __name__ == "__main__":
    main()
