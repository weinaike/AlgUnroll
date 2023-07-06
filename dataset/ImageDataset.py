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

    def __init__(self, path, train=True):
       
        self.images = list()
        self.targets = list()
        self.sample_count = 0
        # path = "/media/ausu-x299/diffuserCam_dataset/"
        if train == True:
            csv_file = os.path.join(path,"dataset_train_short.csv")    
        else:
            csv_file = os.path.join(path,"dataset_test.csv")

        self.csv_data = np.loadtxt(open(csv_file,"rb"),dtype=str,usecols=[0])             
        self.sample_count = len(self.csv_data)

        for item in self.csv_data:
            item = item[:-9]+".npy"
            self.images.append(os.path.join(path,"diffuser_images",item))
            self.targets.append(os.path.join(path,"ground_truth_lensed", item))

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):

        img = np.load(self.images[idx])
        target = np.load(self.targets[idx])

        return torch.tensor(img).float(),  torch.tensor(target).float()


if __name__ == '__main__':
    start = time.time()
    path = "/media/ausu-x299/diffuserCam_dataset/"

    data = ImageFileDataset(path, train=True)
    for i in range(3):
        img, target = data.__getitem__(i)
        fig, axis = plt.subplots(1,2)
        axis[0].imshow(img)
        axis[1].imshow(target)
        plt.savefig("a_{}.png".format(i))
    end = time.time()
    print(end-start)
