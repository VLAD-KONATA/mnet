import torch
from torch import nn
from util_evaluation import calc_ssim
import torch.nn.functional as F

class Select_Loss(nn.Module):
    def __init__(self,args):
        super(Select_Loss,self).__init__()
        self.args = args
        self.l1loss  = nn.L1Loss()
        self.mseloss = nn.MSELoss(reduction='none')
        self.contractloss=MultiContrastLoss()
        self.loss_compact=0.1
        self.loss_separate=0.1


    def forward(self,sr,gt,separateness_loss=None, compactness_loss=None):
        if separateness_loss!=None and compactness_loss!=None:
            mseloss=torch.mean(self.mseloss(sr,gt))
            loss=mseloss+ self.loss_compact * compactness_loss + self.loss_separate * separateness_loss
        else:
            if False:
                l1loss = self.l1loss(sr,gt)
                loss = l1loss 
            else:
                loss=self.contractloss(sr,gt)
        return loss
    
class MultiContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = calc_ssim()  # 需要实现或导入SSIM
        
    def forward(self, pred, gt, features):
        # 基础L1和SSIM损失
        l1 = self.l1_loss(pred, gt)
        ssim = 1 - self.ssim_loss(pred, gt)
        
        # 对比度相关损失
        pred_contrast = self.calc_contrast(pred)
        gt_contrast = self.calc_contrast(gt)
        contrast_loss = F.mse_loss(pred_contrast, gt_contrast)
        
        return l1 + ssim + 0.5*contrast_loss
    
    def calc_contrast(self, x):
        # 计算局部对比度
        avg_pool = F.avg_pool2d(x, 3, stride=1, padding=1)
        return torch.abs(x - avg_pool)