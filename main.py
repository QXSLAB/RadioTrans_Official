import os
import torch
import torch.nn as nn
from math import floor
from torch.optim import Adam
from raw import PowerSet
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from model import CGAN_G, CGAN_D, weight_init

def main():

    trail = "mlp_init"

    experiment = "/home/dell/hdd/program_fsrpe/{}".format(trail)

    # make dir to save result
    if os.path.exists(experiment):
        y = input("folder {}, overwrite? [Y/N]:".format(experiment))
        if y.lower()=="y":
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
    tl = floor(0.8*len(dset))
    tset, vset = random_split(dset, [tl, len(dset)-tl])
    tloader = DataLoader(tset, batch_size=128, 
                            shuffle=True, num_workers=4,
                            drop_last=True)
    vloader = DataLoader(vset, batch_size=128, 
                            shuffle=False, num_workers=4)

    # check data
    fixed, real_batch = next(iter(tloader))
    real_img = make_grid(real_batch, normalize=True)
    writer.add_image("real", real_img)

    # setup loss
    criterion = nn.BCELoss()

    # setup model
    G = CGAN_G().cuda()
    D = CGAN_D().cuda()

    # apply weight init
    G.apply(weight_init)
    D.apply(weight_init)

    # set optimizer
    G_opt = Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    D_opt = Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    # record global step
    step = 0

    for e in range(100):
        
        iter_loader = iter(tloader)
        
        # run until exhausted
        while True:

            # clear grad in D

            D.zero_grad()

            # grad of real: y matched with x

            x, y = next(iter_loader, ("",""))
            if y=="": break
            x, y = x.cuda(), y.cuda()

            o = D(x, y)
            errD_real = criterion(o, o.new_ones(o.shape))
            errD_real.backward()

            # grad of fake: y not matched with x

            _, y = next(iter_loader, ("",""))
            if y=="": break
            y = y.cuda()

            o = D(x, y)
            errD_no_match = criterion(o, o.new_zeros(o.shape))
            errD_no_match.backward()

            # grad of fake: generated y

            z = torch.rand(x.shape[0], 100).cuda()
            y = G(z, x)
            y = y.detach()

            o = D(x, y)
            errD_gen = criterion(o, o.new_zeros(o.shape))
            errD_gen.backward()

            # update D

            D_opt.step()

            # clear grad in G

            G.zero_grad()
            
            # cal grad in G

            x, _ = next(iter_loader, ("",""))
            if x=="": break
            z = torch.rand(x.shape[0], 100)
            z, x = z.cuda(), x.cuda()

            y = G(z, x)
            o = D(x, y)
            errG = criterion(o, o.new_ones(o.shape))
            errG.backward()
            
            step += 1
            if step % 10 == 9: continue

            # track training
            print("Epoch [{}] Global [{}]".format(e, step), end="\r")

            # output in tensorboard 

            errD = {"real": errD_real,
                    "no_match": errD_no_match,
                    "gen": errD_gen,
                    "all": errD_real + errD_no_match + errD_gen}
            writer.add_scalars("err/D", errD, step)

            writer.add_scalar("err/G", errG, step)
            
            z = torch.rand(x.shape[0], 100)
            y = G(z.cuda(), fixed.cuda())
            fake_img = make_grid(y, normalize=True)
            writer.add_image("fake", fake_img, step)

        # save model

        if e%10 == 9:
           torch.save(G.state_dict(), 
                      os.path.join(experiment, "G_{}.pt".format(e)))
           torch.save(D.state_dict(), 
                      os.path.join(experiment, "D_{}.pt".format(e)))
                      
if __name__ == "__main__":
    main()
