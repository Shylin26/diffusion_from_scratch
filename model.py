import torch
import torch.nn as nn
import math


class SimpleDenoiser(nn.Module):
    def __init__(self,time_dim=32):
        super(). __init__()

        self.net=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,1,3,padding=1)
        )
    
    def forward(self,x,t):
       

        return self.net(x)

