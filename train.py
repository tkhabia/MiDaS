import torch.nn as nn
import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader , Dataset
from midas.midas_net_custom import MidasNet_small
import os.path as path
import glob

model = MidasNet_small(None , features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})

transform = transforms = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

class walldata(Dataset):
    def __init__(self , data , transform) -> None:
        super().__init__()
        self.data = data 
        self.transform = transform
    
    def __len__ (self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = cv2.cvtColor( cv2.imread(self.data[index][0]) , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img , ( 256, 256))

        mask = cv2.resize( cv2.imread( self.data[index][1] ,  0 ) , ( 256 , 256 ))
        
        img = self.transform(img)
        mask = self.transform(mask) 
        
        return {"img":img , "mask":mask}

data = []
for i in range(146):
    data.append(["./input/img/{}.png".format(i) , "./input/label/{}.png".format(i)])

datawall = walldata(data , transform) 

optimizer = torch.optim.Adam( model.parameters(), 1e-3 )
l1_criterion = nn.L1Loss()
epoch = 400 
train_loader = DataLoader(datawall) 
# def train (model  , train_loader , val_loader , optimizer  , criterion , epoch):
train_loss = []
val_loss =[]
for ep in range(epoch):
    rloss = 0 ; 
    for train in train_loader :
        optimizer.zero_grad()
        img = torch.autograd.Variable( train['img'].cuda())
        mask =torch.autograd.Variable( train['mask'].cuda())

        output = model(img)
        loss = l1_criterion(output, mask)
        train_loss.append(loss)
        loss.backward()
        optimizer.step()
        



