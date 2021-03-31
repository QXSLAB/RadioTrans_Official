"""
    process raw data
"""

import os
import re
from functools import cmp_to_key
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

def param_norm(raw):

    """
        normalize transmit parameter to [-1,1]

        [input]
        raw: shape (batch, 4), 4 channels contains
             [center frequency, x location, y location, antenna angle]

        [output]
        out: shape (batch, 4)
    """

    minimum = raw.min(axis=0, keepdims=True)
    maximum = raw.max(axis=0, keepdims=True)
    out = (raw-minimum)/(maximum-minimum)
    out = (out-0.5)/0.5

    return out


def to_power_map(source, target, image_size):

    """
        convert received power to image

        [input]
        source: folder saving wireless insite simulation result
        target: folder saving power-map images
        image_size: width and height of simulation grid

        [output]
        power-map saved in target folder
        filename: transmit parameter
        pixel: power on grid
    """

    # make target folder
    if os.path.exists(target):
        yes = input("{} exists, overwrite?[Y/N]".format(target))
        if yes.lower() == "y":
            os.system("trash {}".format(target))
        else:
            exit("make a new folder")
    os.mkdir(target)

    param_file ="test_prj.STA.xml"
    power_file = os.path.join("STA" ,"test_prj.power.t001_01.r002.p2m")
    build_file = "building.dat"
    tree_file = "tree_list.dat"

    for root, dirs, _ in os.walk(source):
        for d in dirs:

            print("processing folder {}".format(d))

            param_path = os.path.join(root, d, param_file)
            power_path = os.path.join(root, d, power_file)
            build_path = os.path.join(root, d, build_file)
            tree_path = os.path.join(root, d, tree_file)

            if not (os.path.exists(param_path) and 
                    os.path.exists(power_path) and
                    os.path.exists(build_path) and
                    os.path.exists(tree_path)):
                continue

            # read tx param
            with open(param_path, 'r') as f:
                l = f.readlines()
                freq = float(re.findall(r'-?\d+\.?\d*', l[497-1])[0])
                x = float(re.findall(r'-?\d+\.?\d*', l[612-1])[0])
                y = float(re.findall(r'-?\d+\.?\d*', l[615-1])[0])
                z = float(re.findall(r'-?\d+\.?\d*', l[618-1])[0])
            param = [freq, x, y, z]
            
            # read rx power
            power, phase = [], []
            with open(power_path, 'r') as f:
                for l in f:
                    if l[0] == '#':
                        continue
                    s = re.findall(r'-?\d+\.?\d*', l)
                    power.append(float(s[-2]))
                    phase.append(float(s[-1]))

            # read landscape
            build_map = np.loadtxt(build_path)
            build_map = np.transpose(build_map)

            tree_list = np.loadtxt(tree_path)
            loc_x, loc_y = np.uint(tree_list[:, 1]), np.uint(tree_list[:, 2])
            tree_map = build_map*0
            tree_map[loc_x, loc_y] = tree_list[:, 3]

            power = np.reshape(power, (image_size, image_size))
            phase = np.reshape(phase, (image_size, image_size))

            #plt.imshow(build_map)
            #plt.scatter(loc_x, loc_y, tree_list[:, 3])
            #plt.scatter(x, y, marker="o", s=500, c='r')
            #plt.savefig(os.path.join(target, "{}-landscape.png".format(param)))
            #plt.close()
            #
            #plt.imshow(power)
            #plt.savefig(os.path.join(target, '{}-power.png'.format(param)))
            #plt.close()

            np.savez(os.path.join(target, "{}.npz".format(param)),
                     build_map, tree_map, param, power, phase)


class PowerSet(Dataset):

    """
        dataset used in pytorch
    """

    def __init__(self, target):

        self.root, _, self.files = list(os.walk(target))[0]

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        x = self.files[index]

        fz = np.load(os.path.join(self.root, x))

        return list(zip(*fz.items()))[1]


if __name__ == '__main__':

    source = '/media/qxs/My Passport/random_scenario_simulation'
    target = '/home/qxs/hdd/random_scen'
    to_power_map(source, target, 64)

    #dset = PowerSet('/home/qxs/hdd/random_scen')
    #dloader = DataLoader(dset)
    #print(next(iter(dloader)))
    #print(next(iter(dloader)))
