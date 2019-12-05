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
# get dataframes
from loaders import train_df , test_df

#%%
# generate triplets

class AttTriplets(Dataset):

    def __init__(self,n_triplets,train_df,test_df):
        self.transform_to_tensor = transforms.ToTensor()
        self.n_triplets = n_triplets
        self.train_df = train_df
        self.test_df = test_df
        self.df = pd.concat([train_df,test_df],axis=0)

        self.labels = self.df.iloc[:,1]
        self.triplets = []

        for x in range(self.n_triplets):
            anchor_idx = np.random.randint(0,self.labels.shape[0])
            anchor_label = self.df.iloc[anchor_idx]['labels']

            pos_idxs = np.where(anchor_label == self.df['labels'])
            pos_idxs = np.array(pos_idxs[0])

            neg_idxs = np.where(anchor_label != self.df['labels'])
            neg_idxs = np.array(neg_idxs[0])

            anchor,positive = np.random.choice(pos_idxs,2,replace=False)
            negative = np.random.choice(neg_idxs,1,replace=False)[0]
            self.triplets.append([anchor,positive,negative])



    def __getitem__(self,index):

        train_triplets = self.triplets

        anchor_image = Image.open(self.df.iloc[train_triplets[index][0]]['path'])
        anchor_image = anchor_image.convert('RGB')
        anchor_tensor = self.transform_to_tensor(anchor_image)

        pos_image = Image.open(self.df.iloc[train_triplets[index][1]]['path'])
        pos_image = pos_image.convert('RGB')
        pos_tensor = self.transform_to_tensor(pos_image)

        neg_image = Image.open(self.df.iloc[train_triplets[index][2]]['path'])
        neg_image = neg_image.convert('RGB')
        neg_tensor = self.transform_to_tensor(neg_image)

        return anchor_tensor,pos_tensor,neg_tensor

    def __len__(self):
        return self.n_triplets

#%%
train_set = AttTriplets(800,train_df,test_df)
train_loader = DataLoader(train_set,batch_size=8,shuffle=True)

test_set = AttTriplets(200,train_df,test_df)
test_loader = DataLoader(test_set,batch_size=8,shuffle=False)

#%%
# Triplet loss
class OldTripletLoss(nn.Module):

    def __init__(self,margin):
        super(TripletLoss,self).__init__()
        self.margin = margin

    def forward(self,anchor,positive,negative):
        dap = ( (anchor - positive)**2 ).sum(1)
        dan = ( (anchor - negative)**2 ).sum(1)
        dist = torch.sum(dap-dan,dim=-1) + self.margin

        hinge = torch.clamp(dist,min=0)
        loss = torch.mean(hinge)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class NewTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
#%%
def triplet_criterion(anc,pos,neg,margin=1.0):
    return TripletLoss(margin)(anc,pos,neg)

#%%
def main():

    d_iter = iter(train_loader)
    a,p,n = d_iter.next()

    print('generated triplets')
    print('train triplets',train_set.__len__())
    print('test triplets',test_set.__len__())

if __name__ == "__main__":
    main()
