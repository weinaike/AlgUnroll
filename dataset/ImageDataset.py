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

class ImageFileDataset(Dataset):
    """Class for getting data as a Dict
    Args:

    Output:
        sample : Dict of images and labels"""

    def __init__(self, sp_file, train=True, data_path = None):
       
        self.dection = None
        self.spectral = None
        
        if train == True:
            self.sample_count = 128
            self.dection = np.load(os.path.join(data_path, "compress.npy"))
            self.spectral = np.load(os.path.join(data_path, "spectral.npy"))
        else:
            self.sample_count = 128
            self.dection = np.load(os.path.join(data_path, "compress_val.npy"))
            self.spectral = np.load(os.path.join(data_path, "spectral_val.npy"))

        self.sp = np.load(sp_file)
        self.length, self.det_num = self.sp.shape

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):

        dection = self.dection[idx]
        spectral = self.spectral[idx]

        return torch.tensor(dection).float(),  torch.tensor(spectral).float()

    def get_size(self):
        return [self.length, self.det_num]


if __name__ == '__main__':
    start = time.time()
    sp_file = "data/SpectralResponse_9.npy"

    data = ImageFileDataset(sp_file, train=True, have_noise = False )
    print(data.get_size())

    det_list = list()
    sp_list = list()
    for i in range(1024):
        det, sp = data.__getitem__(i)
        det_list.append(det.numpy())
        sp_list.append(sp.numpy())
    path = "data/SpectralResponse_9_1024_multi/"
    if not os.path.exists(path):
        os.mkdir(path)
    np.save("{}/compress.npy".format(path), det_list)
    np.save("{}/spectral.npy".format(path), sp_list)

    # print(det)
    # end = time.time()
    # print(end-start)

    # plt.plot(sp)
    # plt.show()