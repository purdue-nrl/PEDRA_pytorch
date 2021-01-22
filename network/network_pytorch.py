# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:48:23 2021

@author: aparna
"""

import torch.nn as nn
import torch
import numpy as np
from network.loss_functions import huber_loss

# this network can be used for reinforce algorithm
class C3F2_with_baseline(nn.Module):
    def __init__(self, num_actions, in_ch=3):
        super(C3F2_with_baseline, self).__init__()
        # feature extractor
        self.conv1 = nn.Conv2d(in_channels = in_ch, kernel_size=7, out_channels=96, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels = 96, kernel_size=5, out_channels=64, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels = 64, kernel_size=3, out_channels=64, stride=1)
        # Main Network
        self.policy = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_actions) ) # add softmax in forward
        # Baseline Network
        self.baseline = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)) # no non-linearity after this layer
        
        
    def forward(self, x):
        
        x = torch.squeeze(x, 0).permute(0,3,1,2)
        #print(np.shape(x))
        x = self.maxpool1(self.conv1(x))
        #print('here')
        x = self.maxpool2(self.conv2(x))
        x = self.conv3(x)
        #print(np.shape(x))
        features = torch.reshape(x, (x.size(0),-1))
        action   = self.policy(features)
        baseline = self.baseline(features)
        
        return action, baseline
    
   
        
        
        

  
