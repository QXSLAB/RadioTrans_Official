"""
    main file
"""

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
    fake = G(param)
    fake = fake.reshape(-1, 1, 64, 64)
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

    trail = "unet_l1_loss"
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

    # check data
    fixed_param, fixed_match = next(iter(vloader))
    fixed_param, fixed_match = fixed_param.cuda(), fixed_match.cuda()
    match_grid = make_grid(fixed_match[:64], normalize=True)
    writer.add_image("match", match_grid)

    # setup model
    G = Unet().cuda()

    # visulize model
    writer.add_graph(G, fixed_param)

    # apply weight init
    G.apply(weight_init)

    # set optimizer
    G_opt = Adam(G.parameters(), lr=1e-4, betas=(0, 0.9))

    # recored best result
    best = float("inf")

    metric = l1_loss

    step = 0

    for epoch in range(100_000):

        # training
        G.train()
        for param, match in tloader:

            param, match = param.cuda(), match.cuda()

            # clear grad in G
            G.zero_grad()

            # grad of G
            fake = G(param)
            train_l1 = l1_loss(fake, match)
            train_mse = mse_loss(fake, match)
            train_loss = metric(fake, match)
            train_loss.backward()

            # update G
            G_opt.step()
            step += 1

            # track progress
            if step % 10 == 9:

                print("Epoch [{:5d}] Global [{:8d}] "
                      "train_loss [{:2.5f}/{:2.5f}] ".format(epoch, step,
                       train_l1, train_mse))

            # visualize performance curve
            if step % 100 == 99:

                writer.add_scalars("err/loss", {"train_l1": train_l1, "train_mse": train_mse}, step)

            # visualize weight, grad
            if step % 1000 == 999:

                # visualize weight and grad
                visualize_weight_grad(writer, G, step)

        # validating
        G.eval()
        with torch.no_grad():
            val_l1, val_mse, val_loss = [], [], []
            for param, match in vloader:
                param, match = param.cuda(), match.cuda()
                fake = G(param)
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
            visualize_gen(G, (fixed_param, fixed_match), metric, experiment)

            # save best model
            torch.save(G.state_dict(),
                       os.path.join(experiment, "G_best.pt"))

        
        print("Epoch [{:5d}] Global [{:8d}] {} "
                "train_loss [{:2.5f}/{:2.5f}] val_loss[{:2.5f}/{:2.5f}]".format(epoch, step,
              metric.__name__, train_l1, train_mse, val_l1, val_mse))
        
        writer.add_scalars("err/loss", {"val_l1": val_l1, "val_mse": val_mse}, step)


if __name__ == "__main__":
    main()
