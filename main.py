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
from model import CGAN_G, CGAN_D, weight_init
from model import C_DCGAN_G, C_DCGAN_D
from model import C_ResNet_G, C_ResNet_D


def setup_seed(seed):

    """
        make result reproducible
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sample_from_data(loader):

    """
        sample dataloader looply
    """

    itera, epoch = iter(loader), 0
    while True:
        try:
            param, match = next(itera)
        except:
            epoch += 1
            itera = iter(loader)
            param, match = next(itera)
        finally:
            yield param.cuda(), match.cuda(), epoch


def sample_from_gen(G, param):

    """
        generate fake data using G
    """

    noise = torch.rand(param.shape[0], 100).cuda()
    fake = G(noise, param)

    return fake


def grad_penalty(D, param, match, fake):

    """
        gradient penalty for wgan-gp
    """

    # grab data from sampling distribution
    epsl = torch.rand(match.shape[0], 1, 1, 1).cuda()
    interp = epsl*match + (1-epsl)*fake
    interp.requires_grad = True

    # grad w.r.t. x in sampling distribution
    flag = D(param, interp)
    grad = autograd.grad(outputs=flag, inputs=interp,
                         grad_outputs=torch.full_like(flag, 1),
                         create_graph=True, retain_graph=True,
                         only_inputs=True)
    grad = grad[0].view(match.shape[0], -1)

    # two side grad penalty
    penalty = ((grad.norm(p=2, dim=1)-1)**2).mean()

    return penalty


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
    fake = sample_from_gen(G, param)
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

    trail = "mse_unet_tanh_bs64"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    # sample train data looply
    train_sampler = sample_from_data(tloader)

    # check data
    fixed, match_batch, _ = next(sample_from_data(vloader))
    match_grid = make_grid(match_batch[:64], normalize=True)
    writer.add_image("match", match_grid)

    # setup model
    # G = CGAN_G().cuda()
    # D = CGAN_D().cuda()
    G = C_ResNet_G().cuda()

    # difine visualization model
    class CGAN(nn.Module):
        """
            for model visualization
        """
        def __init__(self, G):
            super(CGAN, self).__init__()
            self.G = G
        def forward(self, param):
            fake = sample_from_gen(self.G, param)
            return fake

    # visulize model
    cgan = CGAN(G)
    writer.add_graph(cgan, fixed)

    # apply weight init
    G.apply(weight_init)

    # set optimizer
    # TODO change beta1
    G_opt = Adam(G.parameters(), lr=1e-4, betas=(0, 0.9))

    # recored best result
    best = float("inf")

    # setup metric
    metric = l1_loss

    for step in range(100_000):

        # clear grad in G
        G.zero_grad()

        # grab data for G
        param, match, epoch = next(train_sampler)

        # grad of G
        fake = sample_from_gen(G, param)
        errG = mse_loss(fake, match)
        errG.backward()

        # generation quality
        qlty = quality(fake, match, metric)

        # update G
        G_opt.step()

        # adjust lr
        #G_scheduler.step()
        #D_scheduler.step()

        # save best result
        if qlty < best:

            best = qlty

            # visualize best fake
            visualize_gen(G, (fixed, match_batch), metric, experiment)

            # save best model
            torch.save(G.state_dict(),
                       os.path.join(experiment, "G_best.pt"))

        # track progress
        if step % 10 == 9:

            print("Epoch [{:5d}] Global [{:8d}] "
                  "errG [{:2.5f}] {} [{:2.5f}/{:2.5f}]".format(epoch, step,
                   errG, metric.__name__, qlty, best))

        # visualize performance curve
        if step % 100 == 99:

            errD = {"mse": errG}
            errM = {"qlty": qlty,
                    "best": best}
            writer.add_scalars("err/D", errD, step)
            writer.add_scalars("err/{}".format(metric.__name__), errM, step)

        # visualize weight, grad, fake
        if step % 1000 == 999:

            # visualize weight and grad
            visualize_weight_grad(writer, G, step)

            # visualize fake
            visualize_gen(G, (fixed, match_batch), metric,
                          "step {} epoch {}".format(step, epoch), writer)


if __name__ == "__main__":
    main()
