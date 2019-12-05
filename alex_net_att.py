#!/home/adi/anaconda3/bin/python3.7

#%%
# basic imports
import numpy as np
import pandas as pd
import os

# image to open images
from PIL import Image

# torch imports
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.functional as F

# get loaders
from loaders import train_loader , test_loader

#%%
# dummy
dummy_iter = iter(train_loader)
sample_img,sample_lbl = dummy_iter.next()
# print(sample_lbl)
# print(sample_img.shape)

#%%
# choose model
alex_net = torchvision.models.alexnet(pretrained=True)

#%%
# architecture
alex_net

#%%
# dummy through
out = alex_net(sample_img)

#%%
# out.shape

#%%
# freeze model weights and biases
for param in alex_net.parameters():
    param.requires_grad = False

#%%
# only train the classifier
# last layers
alex_net.classifier[-1]

#%%
alex_net.classifier[1] = nn.Linear(in_features=9216,out_features=1024)
alex_net.classifier[4] = nn.Linear(in_features=1024,out_features=512)
alex_net.classifier[-1] = nn.Linear(in_features=512,out_features=128)

#%%
# alex_net.classifier
# to train :

#%%
def main():
    print('Architecture :')
    print(alex_net)
    for param in alex_net.parameters():
        if param.requires_grad :
            # print(param.numel())
            pass
    #%%
    # that is about
    print('9216*1024' )
    print('+','1024')
    print('1024*512')
    print('+','512')
    print('512*128')
    print('+','128')
    print('total params to train :',9437184
    +1024
    +524288
    +512
    +65536
    +128)

if __name__ == "__main__":
    main()
