#!/home/adi/anaconda3/bin/python3.7

#%%
# basic imports
import numpy as np
import pandas as pd
import os

#image to open images
from PIL import Image
# torch imports
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


#%%
def get_dataframe():


    # get path
    source_path = "./data/faces/"
    train_path = source_path + "training/"
    test_path = source_path + "testing/"

    # store images
    testing_images = []
    training_images = []
    # labels
    testing_labels = []
    training_labels = []

    # for testing set
    for folder,subfolder,image in os.walk(test_path):
        for img in image:
            testing_images.append(folder+'/'+str(img))
            testing_labels.append(int(folder.split('/')[-1][1:]))

    # for training set
    for folder,subfolder,image in os.walk(train_path):
        for img in image:
            training_images.append(folder+'/'+str(img))
            training_labels.append(int(folder.split('/')[-1][1:]))



    # make data frame
    # testing_df
    test_dict = dict(path=testing_images,labels=testing_labels)
    train_dict = dict(path=training_images,labels=training_labels)
    test_df = pd.DataFrame(test_dict)
    train_df = pd.DataFrame(train_dict)

    return train_df , test_df


#%%
class AttDataset(Dataset):
    def __init__(self,df):
        # transforms
        self.transform_to_tensor = transforms.ToTensor()
        # DataFrame
        self.df = df
        # path
        self.img_arr = np.asarray(df.iloc[:,0])
        # label
        self.label = np.asarray(df.iloc[:,1],dtype=np.int8)

    def __getitem__(self,index):
        # image and label of that index
        img = self.img_arr[index]
        lbl = self.label[index]

        # open the image
        img = Image.open(img)

        # modern networks only work
        #+ on rgb images
        img = img.convert('RGB')

        # apply the transforms
        tensor_img = self.transform_to_tensor(img)
        tensor_lbl = torch.tensor(lbl)

        # return
        return tensor_img , tensor_lbl

    def __len__(self):
        return self.df.shape[0]



#%%
train_df,test_df = get_dataframe()
train_set = AttDataset(train_df)
test_set = AttDataset(test_df)
train_loader = DataLoader(train_set,batch_size=8,shuffle=True)
test_loader = DataLoader(test_set,batch_size=8,shuffle=False)
