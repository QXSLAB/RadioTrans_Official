import os
import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from math import floor
from torch.optim import Adam
from raw import PowerSet
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from model import CGAN_G, CGAN_D, weight_init


def visulize_weight_grad(writer, model, step):

    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        writer.add_histogram("weight/"+tag, value, step)
        writer.add_histogram("grad/"+tag, value.grad, step)


def main():

    trail = "mlp_epoch1e4"

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
    fixed, real_batch = fixed.cuda(), real_batch.cuda()
    real_img = make_grid(real_batch[:64], normalize=True)
    writer.add_image("real", real_img)

    # setup loss
    criterion = nn.BCELoss()

    # setup model
    G = CGAN_G().cuda()
    D = CGAN_D().cuda()

    # visulize model
    class CGAN(nn.Module):
        def __init__(self, G, D):
            super(CGAN, self).__init__()
            self.G = G
            self.D = D
        def forward(self, x):
            z = torch.rand(x.shape[0], 100).cuda()
            y = self.G(z, x)
            o = self.D(x, y)
            return o
    cgan = CGAN(G,D)
    writer.add_graph(cgan, fixed)

    # apply weight init
    G.apply(weight_init)
    D.apply(weight_init)

    # set optimizer
    G_opt = Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    D_opt = Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    # record global step
    step = 0

    for e in range(10000):
        
        iter_loader = iter(tloader)
        
        # run until exhausted
        while True:

            # grab data for D

            x, match = next(iter_loader, ("",""))
            if match=="": break
            x, match = x.cuda(), match.cuda()


            _, no_match = next(iter_loader, ("",""))
            if no_match=="": break
            no_match = no_match.cuda()
            
            z = torch.rand(x.shape[0], 100).cuda()
            fake = G(z, x).detach()

            # clear grad in D

            D.zero_grad()

            # grad of match

            o = D(x, match)
            errD_match = criterion(o, o.new_ones(o.shape))
            errD_match.backward()

            # grad of no_match

            o = D(x, no_match)
            errD_no_match = criterion(o, o.new_zeros(o.shape))
            errD_no_match.backward()

            # grad of fake

            o = D(x, fake)
            errD_fake = criterion(o, o.new_zeros(o.shape))
            errD_fake.backward()

            # update D

            D_opt.step()


            # grab data for G

            x, gt = next(iter_loader, ("",""))
            if x=="": break
            z = torch.rand(x.shape[0], 100)
            z, x, gt = z.cuda(), x.cuda(), gt.cuda()

            # clear grad in G

            G.zero_grad()
            
            # grad of G

            y = G(z, x)
            o = D(x, y)
            errG = criterion(o, o.new_ones(o.shape))
            errG.backward()

            mae = l1_loss(gt.view(-1,4096), y)
            
            # update G

            G_opt.step()

            
            # update global step

            step += 1
            if step % 10 == 9: continue

            # track progress

            print("Epoch [{}] Global [{}]".format(e, step), end="\r")

            # track performance 

            errD = {"real": errD_match,
                    "no_match": errD_no_match,
                    "gen": errD_fake,
                    "all": errD_match + errD_no_match + errD_fake}
            writer.add_scalars("err/D", errD, step)
            writer.add_scalar("err/G", errG, step)
            writer.add_scalar("err/mae", mae, step)


        # save model

        if e%10 == 9:

            # check weight

            visulize_weight_grad(writer, G, step)
            visulize_weight_grad(writer, D, step)

            # check output

            z = torch.rand(fixed.shape[0], 100).cuda()
            y = G(z, fixed).reshape(-1, 1, 64, 64)
            mae = l1_loss(y, real_batch)
            fake_img = make_grid(y[:64], normalize=True)
            writer.add_image("fake/{0}/{1}".format(e, mae), fake_img)
           
            torch.save(G.state_dict(), 
                        os.path.join(experiment, "G_{}.pt".format(e)))
            torch.save(D.state_dict(), 
                        os.path.join(experiment, "D_{}.pt".format(e)))
                      

if __name__ == "__main__":
    main()
