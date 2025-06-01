import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Memory import *
#from Memory import *

class Encoder(torch.nn.Module):
    def __init__(self, t_length = 2, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        
        return tensorConv4

    
    
class Decoder(torch.nn.Module):
    def __init__(self, t_length = 2, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(512, 512)
        self.moduleUpsample4 = Upsample(512, 512)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 256)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 128)

        self.moduleDeconv1 = Gen(128,n_channel,64)
        
        
        
    def forward(self, x):
        
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        
        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        
        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        
        output = self.moduleDeconv1(tensorUpsample2)

                
        return output
    


class convAE(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 2, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        #self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
       

    def forward(self, x, keys=None,train=True):

        fea = self.encoder(x)
        output=self.decoder(fea)
        return output
        if train:
            #updated_fea= self.memory(fea, keys, train)
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea)
            
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
            #return output
        
        #test
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea)
            
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss

                                          

if __name__=='__main__':
    
    import argparse


    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
    parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

    args = parser.parse_args()
    gpy_id = 0
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    model.cuda()
    m_items = F.normalize(torch.rand((10, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items
    from torchsummary import summary
    summary(model,(3,256, 256),m_items,'True',device='cuda')

    