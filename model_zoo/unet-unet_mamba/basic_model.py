import torch
import torch.nn as nn
from torch.nn import init
import einops
from functools import partial
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba
#from unet import UNet
from .unet import UNet


def make_model(args):
    return NewUNet(4,7,64)

#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MUBlock(nn.Module):
    def __init__(
        self, channels,shape):
        super(MUBlock, self).__init__()
        #self.res_scale = res_scale
        self.unet=UNet(channels,channels)
        self.mamba= Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=shape, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    )

    def forward(self, x):
        x_u=self.unet(x)
       # x_inter = self.inter_slice_branch(x).mul(self.res_scale)
        mamba_x=einops.rearrange(x,'b d h w -> (b d) h w')
        mamba_x=mamba_x.contiguous()
        output = self.mamba(mamba_x)
        x_mamba=einops.rearrange( output,'(b d) h w -> b d h w',b=x.shape[0])
        #out = x_inter + x_mamba + x
        out = x_u + x_mamba + x
        return out

class I2Group(nn.Module):
    def __init__(
        self, conv, n_depth, n_feat, kernel_size,skip_connect=False,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,head_num=1,win_num_sqrt=16,window_size=16):
        super().__init__()

        body = [MUBlock(conv, n_feat, kernel_size,
                            bias, bn, act, res_scale,head_num,win_num_sqrt) for _ in range(n_depth)]

        self.body = nn.Sequential(*body)
    def forward(self,x):
        x_f = self.body(x)
        out = x_f
        return out



    
class NewUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=7, init_features=64):
        super(NewUNet, self).__init__()

        features = init_features
        self.mu1=MUBlock(features,256)
        #self.mu1=MUBlock(in_channels,256)
        self.encoder1 = UNet._block( features, features, name="enc1")
        #self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mu2=MUBlock(features,128)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mu3=MUBlock(features * 2,64)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mu4=MUBlock(features * 4,32)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.um4=MUBlock(features * 8,32)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.um3=MUBlock(features * 4,64)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.um2=MUBlock(features * 2,128)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        self.um1=MUBlock(features ,256)
        
        self.head = nn.Sequential(nn.Conv2d(in_channels, features, kernel_size=3,padding=(3 // 2), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(features, features, kernel_size=3,padding=(3 // 2), bias=True))
        modules_tail = [
            nn.Conv2d(features, features, kernel_size=3,padding=(3 // 2), bias=True),
            nn.ReLU(),
            nn.Conv2d(features, out_channels, kernel_size=3,padding=(3 // 2), bias=True)]
        self.tail = nn.Sequential(*modules_tail)
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x,a=None):
        x=x.permute(0,3,1,2)
        x=x.contiguous()

        x_head = self.head(x) 

        mu1=self.mu1(x_head)
        enc1 = self.encoder1(mu1)
        pool1=self.pool1(enc1)
        mu2=self.mu2(pool1)
        enc2 = self.encoder2(mu2)
        pool2=self.pool2(enc2)
        mu3=self.mu3(pool2)
        enc3 = self.encoder3(mu3)
        pool3=self.pool3(enc3)
        mu4=self.mu4(pool3)
        enc4 = self.encoder4(mu4)
        pool4=self.pool4(enc4)

        bottleneck = self.bottleneck(pool4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4=self.um4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3=self.um3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2=self.um2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1=self.um1(dec1)

        tail=dec1+x_head
        out= self.tail(tail)
        #out= self.conv(dec1)
        out=out.permute(0,2,3,1).contiguous()

        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )
        



if __name__ == '__main__':
    '''
    import argparse
    args = argparse.Namespace()
    args.upscale = 2
    args.n_feats = 64
    args.kernel_size = 3
    args.res_scale = 1
    args.num_blocks = 16
    args.lr_slice_patch = 4
    args.hr_slice_patch = (args.lr_slice_patch-1)*args.upscale + 1
    args.head_num = 1
    args.win_num_sqrt = 16
    args.window_size = 16
    args.n_size = 256
    '''
    gpy_id = 0
    model = NewUNet(4,7,64).cuda(gpy_id)
    from torchsummary import summary
    summary(model,(256, 256, 4))
    #pred=model(x)
    #print(pred.shape)