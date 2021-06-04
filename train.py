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
import random
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
model = MidasNet_small(None , features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})

transform = transforms = transforms.Compose(
    [
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
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
        img = cv2.resize(img , ( 256, 256 ))

        mask = cv2.resize( cv2.imread( self.data[index][1] ,  0 ) , ( 256 , 256 ))
        
        img = self.transform(img)
        mask = mask/255
        
        return {"img":img , "mask":mask}

data = []
for i in range(1, 1409):
    data.append(["./input/img/{}.png".format(i) , "./input/label/{}.png".format(i)])

random.shuffle(data)
datawall = walldata(data[:len(data)*8//10] , transform) 
valwall =  walldata(data[len(data)*8//10 : ] , transform) 

optimizer = torch.optim.Adam( model.parameters(), 1e-3 )
criterion = nn.L1Loss()
epoch = 50
train_loader = DataLoader(datawall ,batch_size=16 , shuffle=True , num_workers=2) 
val_loader = DataLoader(valwall ,batch_size=16  , num_workers=2) 

scheduler = StepLR(optimizer, step_size=2, gamma=0.85)

def train (model  , train_loader , val_loader  , optimizer  , criterion , epoch , scheduler):
    train_loss = []
    val_loss =[]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for ep in range(epoch):
        scheduler.step()
        rloss = 0 
        rvloss= 0 
        for train in tqdm( train_loader) :

            optimizer.zero_grad()
            img = torch.autograd.Variable( train['img'].to(device))
            mask =torch.autograd.Variable( train['mask'].to(device))

            output = model(img)
            loss = criterion(output, mask)
            rloss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss.append(rloss/len(train_loader))

        if ep %4 == 0 :
            with torch.no_grad():
                for train in val_loader:
                    img = torch.autograd.Variable( train['img'].to(device))
                    mask =torch.autograd.Variable( train['mask'].to(device))
                    output = model(img)
                    loss_t = criterion(output, mask)
                    rvloss+= loss_t.item()
                val_loss.append(rvloss)
            print("loss " , loss , "val loss = " , loss_t)
            
        torch.save(model.state_dict(), "../drive/MyDrive/model2.pt")
    plt.plot(loss , label="loss")
    plt.plot(loss_t , label="Val Loss")
    plt.legend()
    plt.savefig("out.png")
    plt.show()

train (model  , train_loader , val_loader , optimizer  , criterion , epoch, scheduler)

