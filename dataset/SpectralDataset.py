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
    def __init__(self, sp_file, train=True, have_noise = False, sig = [200,500], peak_num = 1):
       
        self.sample_count = 256
        if train == False:
            self.sample_count = 128

        self.sp = np.load(sp_file)
        self.length, self.det_num = self.sp.shape
        self.x = np.linspace(355,3735,self.length)
        self.sig_min = sig[0]
        self.sig_max = sig[1]
        self.have_noise = have_noise
        self.detect , self.spectral = self.generate_random(self.sample_count, peak_num)


    def generate_random(self, sample_count, num = 1):
        detect_list = list()
        spectral_list = list()
        if self.have_noise:
            num = random.choice(range(num)) + 1
        for i in range(sample_count):
            spectral = np.zeros(self.x.shape)
            rn = random.choice([0,1]) #随机生成一半的相邻峰
            rn = 1
            if rn > 0:
                center = random.uniform(700,3200)
                sigma = random.uniform(self.sig_min,self.sig_max)
                for j in range(num):                                        
                    amp = random.uniform(0.2,1)
                    amp = 1
                    spectral += amp * np.exp(-1 * (self.x - center)**2 / (sigma**2 ))
                    center = center + sigma * 2.5
            else:
                for j in range(num):
                    center = random.uniform(400,3500)
                    sigma = random.uniform(self.sig_min,self.sig_max)
                    amp = random.uniform(0.2,1)
                    spectral += amp * np.exp(-1 * (self.x - center)**2 / (sigma**2 ))
            spectral = spectral / np.max(spectral) 
            detect = np.matmul(spectral, self.sp)

            detect = torch.tensor(detect).float()
            spectral = torch.tensor(spectral).float()

            detect_list.append(detect)
            spectral_list.append(spectral)
        return detect_list, spectral_list

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        return self.detect[idx], self.spectral[idx]

    def get_size(self):
        return [self.length, self.det_num]


if __name__ == '__main__':
    start = time.time()
    sp_file = "data/SpectralResponse_9.npy"

    data = SpectralDataset(sp_file, train=True, have_noise = False, sig = [100,200] )
    print(data.get_size())

    det_list = list()
    sp_list = list()
    for i in range(256):
        det, sp = data.__getitem__(i)
        det_list.append(det.numpy())
        sp_list.append(sp.numpy())
    path = "data/SpectralResponse_9_1024_multi/"
    if not os.path.exists(path):
        os.mkdir(path)
    np.save("{}/compress_val.npy".format(path), det_list)
    np.save("{}/spectral_val.npy".format(path), sp_list)

    # print(det)
    # end = time.time()
    # print(end-start)

    # plt.plot(sp)
    # plt.show()