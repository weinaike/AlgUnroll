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
            csv_file = os.path.join(path,"dataset_train.csv")    
        else:
            csv_file = os.path.join(path,"dataset_test.csv")

        self.csv_data = np.loadtxt(open(csv_file,"rb"),dtype=str,usecols=[0])             

        for item in self.csv_data:
            if "npy" not  in item:
                if "tiff" not in item:
                    print(item, "is not exist")
                    continue
                item = item[:-9]+".npy"
            img_file = os.path.join(path,"diffuser_images",item)
            gt_file = os.path.join(path,"ground_truth_lensed", item)
            if os.path.exists(img_file) and os.path.exists(gt_file):
                self.images.append(img_file)
                self.targets.append(gt_file)
            else:
                print(item, "is not exists")

        self.sample_count = len(self.images)
    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):

        img = np.load(self.images[idx])
        target = np.load(self.targets[idx])
        h, w, c = img.shape
        for i in range(c):
            img[:,:,i] /= np.linalg.norm(img[:,:,i].ravel())
            target[:,:,i] /= np.linalg.norm(target[:,:,i].ravel())
            # np.save("ldata_{}.npy".format(i), img[:,:,i])

        return torch.tensor(img).float().permute(2,0,1),  torch.tensor(target).float().permute(2,0,1)


if __name__ == '__main__':
    start = time.time()
    path = "/media/ausu-x299/diffuserCam_dataset/"

    data = ImageFileDataset(path, train=True)
    for i in range(3):
        img, target = data.__getitem__(i)
        print(img.size())
        print(target.size())
        fig, axis = plt.subplots(1,2)
        axis[0].imshow(img)
        axis[1].imshow(target)
        #plt.savefig("a_{}.png".format(i))
    end = time.time()
    print(end-start)
