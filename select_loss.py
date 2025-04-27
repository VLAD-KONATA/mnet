import torch
from torch import nn

class Select_Loss(nn.Module):
    def __init__(self,args):
        super(Select_Loss,self).__init__()
        self.args = args
        self.l1loss  = nn.L1Loss()
        self.mseloss = nn.MSELoss(reduction='none')
        self.loss_compact=0.1
        self.loss_separate=0.1


    def forward(self,sr,gt,separateness_loss=None, compactness_loss=None):
        if separateness_loss!=None and compactness_loss!=None:
            mseloss=torch.mean(self.mseloss(sr,gt))
            loss=mseloss+ self.loss_compact * compactness_loss + self.loss_separate * separateness_loss
        else:
            l1loss = self.l1loss(sr,gt)
            loss = l1loss 
        return loss