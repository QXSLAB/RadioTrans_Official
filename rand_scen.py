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
import torchvision.transforms.functional as F

matplotlib.use('Agg')


def preprocess(build, tree, param, power, phase):

    """
        [input]
        build 0~100 -> 0~255
        tree 0~50 -> 0~255
        z: 30~80 -> 0~255
        f: 5735000000~5825000000 -> 0~255

        [output]
        power: -250~-70 -> 0~255
        phase: -180~180 -> 0~255
    """

    f, x, y, z = param

    build = build/100
    build = np.floor(build*255)
    build = np.expand_dims(build, axis=0)
    # shape 1 x 1000 x 1000

    tree = tree/50
    tree = np.floor(tree*255)
    tree = np.expand_dims(tree, axis=0)
    # shape 1 x 1000 x 1000

    z = (z-30)/50
    z = np.floor(z*255)
    x_idx = int(x)-1
    y_idx = int(y)-1
    source = build*0
    source[0, x_idx, y_idx] = z
    # shape 1 x 1000 x 1000
    
    f = (f-5735000000)/90000000
    f = np.floor(f*255)
    freq = build*0+1
    freq = freq*f

    land = np.concatenate([build, tree, source, freq], axis=0)
    land = np.uint8(land.swapaxes(0, 2))

    power = (power+250)/180
    power = np.floor(power*255)
    power = np.uint8(power)

    phase = (phase+180)/360
    phase = np.floor(phase*255)
    phase = np.uint8(phase)

    return land, power, phase

def to_power_map(source, target, image_size):

    """
    """

    # make target folder
    if os.path.exists(target):
        yes = input("{} exists, overwrite?[Y/N]".format(target))
        if yes.lower() == "y":
            os.system("trash {}".format(target))
        else:
            exit("make a new folder")
    os.mkdir(target)
    os.mkdir(os.path.join(target, "land"))
    os.mkdir(os.path.join(target, "power"))
    os.mkdir(os.path.join(target, "phase"))

    param_file ="test_prj.STA.xml"
    power_file = os.path.join("STA" ,"test_prj.power.t001_01.r002.p2m")
    build_file = "building.dat"
    tree_file = "tree_list.dat"

    for root, dirs, _ in os.walk(source):

        dirs = sorted(dirs, key=lambda x: int(x))

        for d in dirs:

            #if not int(d)==169281:
            #    continue

            #import pdb
            #pdb.set_trace()

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
            tree_list = np.loadtxt(tree_path)

            coords = tree_list[:, 1:3]
            anomaly = np.sum(coords>=1000) + np.sum(coords<0)
            if anomaly>0:
                continue            
            loc_x, loc_y = np.uint(tree_list[:, 1]), np.uint(tree_list[:, 2])
            tree_map = build_map*0
            tree_map[loc_x, loc_y] = tree_list[:, 3]

            if not len(power)==64*64:
                continue

            power = np.reshape(power, (image_size, image_size))
            phase = np.reshape(phase, (image_size, image_size))
            
            #build_map = np.transpose(build_map)
            #plt.imshow(build_map)
            #plt.scatter(loc_x, loc_y, tree_list[:, 3])
            #plt.scatter(x, y, marker="o", s=500, c='r')
            #plt.savefig(os.path.join(target, "{}-landscape.png".format(param)))
            #plt.close()
            #
            #plt.imshow(power)
            #plt.savefig(os.path.join(target, '{}-power.png'.format(param)))
            #plt.close()
            
            clip_ano = np.sum(build_map>=100) + \
                       np.sum(tree_map>=50) + \
                       np.sum(power>-70)

            if clip_ano:
                continue

            land, power, phase = preprocess(build_map, tree_map, param, power, phase)

            land_img = Image.fromarray(land, mode='RGBA')
            land_img.save(os.path.join(target, "land", "{}-{}.png".format(d, param)))

            power_img = Image.fromarray(power)
            power_img.save(os.path.join(target, "power", "{}-{}.png".format(d, param)))

            phase_img = Image.fromarray(phase)
            phase_img.save(os.path.join(target, "phase", "{}-{}.png".format(d, param)))

            return land, power, phase


class PowerSet(Dataset):

    """
        dataset used in pytorch
    """

    def __init__(self, target):

        self.land_path = os.path.join(target, "land")
        self.power_path = os.path.join(target, "power")
        self.phase_path = os.path.join(target, "phase")
        _, _, self.files = list(os.walk(self.land_path))[0]

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        x = self.files[index]

        land = os.path.join(self.land_path, x)
        land = Image.open(land)
        land = F.to_tensor(land)

        power = os.path.join(self.power_path, x)
        power = Image.open(power)
        power = F.to_tensor(power)

        phase = os.path.join(self.phase_path, x)
        phase = Image.open(phase)
        phase = F.to_tensor(phase)

        return land, power, phase


if __name__ == '__main__':

    #source = '/media/qxs/My Passport/random_scenario_simulation'
    #target = '/home/qxs/hdd/test'
    #to_power_map(source, target, 64)
    
    land, power, phase = to_power_map("/home/qxs/hdd/test",
                 "/home/qxs/hdd/test_res", 64)
    dset = PowerSet('/home/qxs/hdd/test_res')
    dloader = DataLoader(dset, batch_size=1)
    l_t, po_t, ph_t =  next(iter(dloader))
    import pdb
    pdb.set_trace()
