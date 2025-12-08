import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss as MAELoss
from torch.nn import *

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
    
    def reconstruction_loss(self, y_hat, y):
        loss = MSELoss(reduction="sum")(y_hat, y)
        return loss
    
    def forward(self, y_hat, y, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = self.reconstruction_loss(y_hat, y) + kl_div
        return loss