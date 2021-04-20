"""
    main file
"""

import os
import time
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
from rand_scen import PowerSet
from model import Trans, weight_init
import torchvision
import torchvision.transforms.functional as F
from tqdm import tqdm


def setup_seed(seed):

    """
        make result reproducible
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def quality(img, match, metric):

    """
        evaluate generated image
    """

    return metric(img.view(-1), match.view(-1))


def display_quality(grid, attr):

    """
        display quality in image grid
    """

    grid_np = grid.detach().cpu().numpy()
    grid_np = np.uint8(grid_np.transpose(1, 2, 0)*255)
    grid_pil = Image.fromarray(grid_np)
    draw = ImageDraw.Draw(grid_pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/"
                              "dejavu/DejaVuSansMono.ttf", size=30)
    draw.text((0, 0), "{0}".format(attr),
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
    fake = G(param)
    fake = fake.reshape(-1, 1, 64, 64)
    fake_grid = make_grid(fake[:64])

    # compare with match image
    match_grid = make_grid(match[:64])
    diff_grid = torch.abs(fake_grid - match_grid)

    # quality annotation
    qlty = quality(fake, match, metric)
    fake_pil = display_quality(fake_grid, "{0} {1:0.5f}".format(metric.__name__, qlty))
    diff_pil = display_quality(diff_grid, "{0} {1:0.5f}".format(metric.__name__, qlty))
    match_pil = display_quality(match_grid, "{0} {1:0.5f}".format(metric.__name__, qlty))

    if not writer:
        fake_pil.save(os.path.join(msg, "best.png"))
        diff_pil.save(os.path.join(msg, "diff.png"))
        match_pil.save(os.path.join(msg, "match.png"))
    else:
        fake_np = np.asarray(fake_pil).transpose(2, 0, 1)
        writer.add_image("fake/{}".format(msg), fake_np)

        diff_np = np.asarray(diff_pil).transpose(2, 0, 1)
        writer.add_image("diff/{}".format(msg), diff_np)


def display_fixed(fixed, path):

    no, land, match = fixed

    y = land[:, :3, :, :]*0
    for i, x in enumerate(land):
        freq = x[3, 0, 0]
        x = x[:3, :, :]
        pil = display_quality(x, "no:{0}, freq:{1:0.5f}".format(no[i].item(), freq.item()))
        y[i] = F.to_tensor(pil)
    
    torchvision.utils.save_image(y, os.path.join(path, "inp.png"), pad_value=1)
    torchvision.utils.save_image(match, os.path.join(path, "origin.png"))


def land_blur(land):

    land[:, [1], :, :] = 100*F.gaussian_blur(land[:, [1], :, :], (11,11))
    land[:, [2], :, :] = 500*F.gaussian_blur(land[:, [2], :, :], (21,21))

    return land


def main():

    trail = "random_trans_grid"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    experiment = "/home/dell/hdd/program_fsrpe/{}".format(trail)

    # make dir to save result
    if os.path.exists(experiment):
        yes = input("folder {}, overwrite? [Y/N]:".format(experiment))
        if yes.lower() == "y":
            os.system("trash {}".format(experiment))
        else:
            exit("make new folder")
    os.mkdir(experiment)

    # make result reproducible
    setup_seed(99)

    # setup tensorboard
    writer = SummaryWriter(experiment)

    # load data
    dset = PowerSet("/home/qxs/ssd/rand_scen_fix_lose")
    train_l = floor(0.99*len(dset))
    tset, vset = random_split(dset, [train_l, len(dset)-train_l])
    tloader = DataLoader(tset, batch_size=64,
                         shuffle=True, num_workers=4,
                         drop_last=True)
    vloader = DataLoader(vset, batch_size=64,
                         shuffle=False, num_workers=4)

    # check data
    no, land, power, phase =  next(iter(vloader))
    fixed_inp, fixed_match = land.cuda(), power.cuda()
    fixed_inp = land_blur(fixed_inp)
    display_fixed((no, fixed_inp, fixed_match), experiment)

    # setup model
    G = Trans(512).cuda()

    # visulize model
    writer.add_graph(G, fixed_inp)

    # apply weight init
    G.apply(weight_init)

    # set optimizer
    G_opt = Adam(G.parameters(), lr=1e-4, betas=(0, 0.9))

    # test time
    t = [0]
    t.append(time.time())

    loader = iter(tloader)
    t.append(time.time())

    no, land, power, phase = next(loader)
    t.append(time.time())

    inp, match = land.cuda(), power.cuda()
    inp = land_blur(inp)
    t.append(time.time())

    fake = G(inp)
    t.append(time.time())

    loss = l1_loss(fake, match)
    t.append(time.time())

    loss.backward()
    t.append(time.time())

    G_opt.step()
    t.append(time.time())

    start = np.array(t[:-1])
    end = np.array(t[1:])

    print(end-start)

    # recored best result
    best = float("inf")

    metric = l1_loss

    step = 0

    for epoch in range(100_000):

        # training
        G.train()
        for idx, (_, land, power, phase) in enumerate(tloader):

            inp, match = land.cuda(), power.cuda()
            inp = land_blur(inp)

            # clear grad in G
            G.zero_grad()

            # grad of G
            fake = G(inp)
            train_l1 = l1_loss(fake, match)
            train_mse = mse_loss(fake, match)
            train_loss = metric(fake, match)
            train_loss.backward()

            # update G
            G_opt.step()
            step += 1

            # track progress
            if step % 10 == 9:

                print("Epoch [{:5d}/{:5d}/{:5d}] Global [{:8d}] "
                      "train_loss [{:2.5f}/{:2.5f}] ".format(epoch, idx, len(tloader), step,
                       train_l1, train_mse))

            # visualize performance curve
            if step % 100 == 99:

                writer.add_scalars("err/loss",
                                   {"train_l1": train_l1, "train_mse": train_mse}, step)

            # visualize weight, grad
            if step % 1000 == 999:

                # visualize weight and grad
                visualize_weight_grad(writer, G, step)

            if step % 500 ==499:

                # validating
                G.eval()
                with torch.no_grad():
                    val_l1, val_mse, val_loss = [], [], []
                    for _, land, power, phase in tqdm(vloader):
                        inp, match = land.cuda(), power.cuda()
                        inp = land_blur(inp)
                        fake = G(inp)
                        val_l1.append(l1_loss(fake, match).item())
                        val_mse.append(mse_loss(fake, match).item())
                        val_loss.append(metric(fake, match).item())
                    val_l1= sum(val_l1)/len(val_l1)
                    val_mse= sum(val_mse)/len(val_mse)
                    val_loss= sum(val_loss)/len(val_loss)

                # save best result
                if val_loss < best:

                    best = val_loss

                    # visualize best fake
                    visualize_gen(G, (fixed_inp, fixed_match), metric, experiment)

                    # save best model
                    torch.save(G.state_dict(),
                               os.path.join(experiment, "G_best.pt"))

                
                print("Epoch [{:5d}/{:5d}/{:5d}] Global [{:8d}] {} "
                      "train_loss [{:2.5f}/{:2.5f}] val_loss[{:2.5f}/{:2.5f}]".format(
                      epoch, idx, len(tloader), step, metric.__name__,
                      train_l1, train_mse, val_l1, val_mse))
                
                writer.add_scalars("err/loss", {"val_l1": val_l1, "val_mse": val_mse}, step)


if __name__ == "__main__":
    main()
