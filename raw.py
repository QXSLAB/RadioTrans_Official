import os
import re
import torch
from PIL import Image
import numpy as np
from functools import cmp_to_key
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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
        y = input("{} exists, overwrite?[Y/N]".format(target))
        if y.lower()=="y":
            os.system("trash {}".format(target))
        else:
            exit("make a new folder")
    os.mkdir(target)
    
    # read transmit parameter
    param = []
    with open(os.path.join(source, 'parameter.log'), 'r') as f:
        for l in f:
            if l.find('simuTimes') >= 0:
                sample = []
                param.append(sample)
                continue
            numbers = re.findall(r'-?\d+\.?\d*', l)
            sample.extend(map(float, numbers))
    param = np.array(param)
    param = param_norm(param[:,:4])

    # read recieved power
    root, dirs, _ = list(os.walk(os.path.join(source, 'out')))[0]
    # sort in simulation order
    for d in sorted(dirs, key=cmp_to_key(lambda x,y: float(x)-float(y))):
            
        print("processing folder {}".format(d))

        # select power file
        _, _, files = list(os.walk(os.path.join(root, d)))[0]
        fn = list(filter(lambda f: f.find("power")>=0, files))[0]

        power = []
        with open(os.path.join(root, d, fn), 'r') as f:
            for l in f:
                if l[0] == '#':
                    continue
                s = re.findall(r'-?\d+\.?\d*', l)[-2]
                power.append(float(s))
        power = (np.array(power)+250)
        power = np.reshape(power, (image_size, image_size))
        im = Image.fromarray(np.uint8(power))
        im.save(os.path.join(target, '{}.png'.format(param[int(d)-1])))


class PowerSet(Dataset):

    """
        dataset used in pytorch 
    """

    def __init__(self, target, transform):
        
        self.root, _, self.files = list(os.walk(target))[0]
        self.transform = transform


    def __len__(self):
        
        return len(self.files)


    def __getitem__(self, index):

        x = self.files[index]

        y = Image.open(os.path.join(self.root, x))
        y = self.transform(y)

        x = re.findall(r"-?\d+\.?\d*", x)
        x = list(map(float, x))
        x = torch.tensor(x)
        
        return x, y


if __name__ == '__main__':

    #source = '/home/dell/hdd/space_effect_raw/112112'
    #target = '/home/dell/hdd/space_effect_png'
    #to_power_map(source, target, 112)

    dset = PowerSet("/home/dell/hdd/space_effect_png",
                    transforms.ToTensor())
    dloader = DataLoader(dset)
    print(next(iter(dloader)))
    print(next(iter(dloader)))
