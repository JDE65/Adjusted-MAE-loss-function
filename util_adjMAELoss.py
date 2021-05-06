# -*- coding: utf-8 -*-
"""
Created on Sat, Jan 23 23:16:50 2021

Adjusted MAEloss functions for DeepNN, LSTM, CNN and ResNet 

@author: JDE65 (Github)
j.dessain@navagne.com   ///  j.dessain@ieseg.fr
www.navagne.com
All rights reserved - Copyright Navagne (2021)
"""
# ====  PART 0. Installing libraries ============

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjMSELoss(nn.Module):
    def __init__(self):
        super(AdjMSELoss, self).__init__()
                
    def forward(self, outputs, labels):
        loss = (outputs - labels)**2
        adj_fact = torch.mean(torch.abs(labels)) ** 2
        adj = torch.exp(-outputs * labels / adj_fact) 
        #adj = torch.exp(-outputs * labels *1000) 
        loss = loss * adj
        return torch.mean(loss)