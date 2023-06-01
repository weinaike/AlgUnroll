#coding=utf-8

import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SpectralDataset(Dataset):
    """Class for getting data as a Dict
    Args:

    Output:
        sample : Dict of images and labels"""

    def __init__(self, sp_file, train=True, have_noise = True, sig = [200,500]):
       
        self.sample_count = 8192
        if train == False:
            self.sample_count = 256
        
        self.sp = np.load(sp_file)
        self.length, self.det_num = self.sp.shape
        self.x = np.linspace(355,3735,self.length)
        self.sig_min = sig[0]
        self.sig_max = sig[1]

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):

        center = random.uniform(400,3500)
        # center = random.uniform(1000,1100)
        sigma = random.uniform(self.sig_min,self.sig_max)
        
        spectral = np.exp(-1 * (self.x - center)**2 / (sigma**2 ))
        dection = np.matmul(spectral, self.sp)
        dection = dection / np.max(dection)

        return torch.tensor(dection).float(),  torch.tensor(spectral).float()

    def get_size(self):
        return [self.length, self.det_num]


class MultiSpectralDataset(Dataset):
    """Class for getting data as a Dict
    Args:

    Output:
        sample : Dict of images and labels"""

    def __init__(self, sp_file, train=True, have_noise = True, sig = [200,500]):
       
        self.sample_count = 8192
        if train == False:
            self.sample_count = 256
        
        self.sp = np.load(sp_file)
        self.length, self.det_num = self.sp.shape
        self.x = np.linspace(355,3735,self.length)
        self.sig_min = sig[0]
        self.sig_max = sig[1]

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        spectral = np.zeros(self.x.shape)

        for i in range(2):

            center = random.uniform(400,3500)
            # center = random.uniform(1000,1100)
            sigma = random.uniform(self.sig_min,self.sig_max)
            amp = random.uniform(0.2,1)
            spectral += amp * np.exp(-1 * (self.x - center)**2 / (sigma**2 ))
        spectral = spectral / np.max(spectral) 
        dection = np.matmul(spectral, self.sp)
        dection = dection / np.max(dection)

        return torch.tensor(dection).float(),  torch.tensor(spectral).float()

    def get_size(self):
        return [self.length, self.det_num]


if __name__ == '__main__':
    start = time.time()
    sp_file = "data/SpectralResponse_9.npy"

    data = SpectralDataset(sp_file, train=True, have_noise = False )
    print(data.get_size())
    det, sp = data.__getitem__(0)
    print(det)
    end = time.time()
    print(end-start)

    plt.plot(sp)
    plt.show()