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
    return newmodel(args)

#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class I2Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,head_num=1,win_num_sqrt=16,window_size=16):
        super(I2Block, self).__init__()
        inter_slice_branch = [
            nn.PixelUnshuffle(2),
            nn.Conv2d(4*n_feat,4*n_feat,3,1,1),
            nn.ReLU(),
            nn.Conv2d(4*n_feat,4*n_feat,3,1,1), # +
            nn.PixelShuffle(2), # +
            nn.Conv2d(n_feat,n_feat,1,1,0)
        ]
        self.inter_slice_branch = nn.Sequential(*inter_slice_branch)
        self.res_scale = res_scale
        self.unet=UNet(64,64)
        self.mamba= Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=256, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    )

    def forward(self, x):
        #x_u=self.unet(x)
       # x_inter = self.inter_slice_branch(x).mul(self.res_scale)
        mamba_x=einops.rearrange(x,'b d h w -> (b d) h w')
        mamba_x=mamba_x.contiguous()
        output = self.mamba(mamba_x)
        x_mamba=einops.rearrange( output,'(b d) h w -> b d h w',b=x.shape[0])
        #out = x_inter + x_mamba + x
        #out = x_u + x_mamba + x
        out=x_mamba+x
        return out

class I2Group(nn.Module):
    def __init__(
        self, conv, n_depth, n_feat, kernel_size,skip_connect=False,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,head_num=1,win_num_sqrt=16,window_size=16):
        super().__init__()

        body = [I2Block(conv, n_feat, kernel_size,
                            bias, bn, act, res_scale,head_num,win_num_sqrt) for _ in range(n_depth)]

        self.body = nn.Sequential(*body)
    def forward(self,x):
        x_f = self.body(x)
        out = x_f
        return out


class CrossViewBlock(nn.Module):
    def __init__(self,n_feat):
        super().__init__()

        self.norm = nn.LayerNorm(n_feat)

        self.conv_sag = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,1,1,0),
            Rearrange('b c h w -> b h c w'),
            nn.PixelShuffle(2),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.PixelUnshuffle(2),
            Rearrange('b h c w -> b c h w'),
        )
        
        self.conv_cor = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,1,1,0),
            Rearrange('b c h w -> b w c h'),
            nn.PixelShuffle(2),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.PixelUnshuffle(2),
            Rearrange('b w c h -> b c h w'),
        )

    def forward(self,x):
        B,C,H,W = x.shape
        x = einops.rearrange(x,'b c h w -> b (h w) c')
        x = self.norm(x)
        x = einops.rearrange(x,'b (h w) c -> b c h w',h=H,w=W)

        x_sag_f = self.conv_sag(x) # b c h w
        x_cor_f = self.conv_cor(x) # b c h w
        x_out = x_cor_f + x_sag_f
        return x_out
    

class newmodel(nn.Module):
    def __init__(self,args=None,conv=default_conv):
        super(newmodel, self).__init__()
        self.args = args
        
        n_feats = args.n_feats #64
        kernel_size = args.kernel_size # 3
        num_blocks = args.num_blocks # 16
        act = nn.ReLU(True)
        res_scale = args.res_scale # 1
        in_slice = args.lr_slice_patch*1
        out_slice = args.hr_slice_patch

        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        window_size = args.window_size
        self.head = nn.Sequential(conv(in_slice,n_feats,kernel_size),
                                  nn.ReLU(),
                                  conv(n_feats,n_feats,kernel_size))
        
        modules_body = [
            I2Group(
                conv, n_depth=1,n_feat=n_feats, kernel_size=kernel_size, act=act, res_scale=res_scale, 
                head_num=head_num, win_num_sqrt=win_num_sqrt,window_size=window_size) for _ in range(2//2)]
        self.body = nn.ModuleList(modules_body)
        
        self.alignment = nn.ModuleList([CrossViewBlock(n_feats) for _ in range(2)])

        self.fuse_align = nn.Conv2d(2*n_feats,n_feats,1,1,0)

        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats,out_slice,kernel_size)]
        self.tail = nn.Sequential(*modules_tail)
        
    def forward(self, x,train=True):
        x = x.permute(0,3,1,2)
        x = x.contiguous()
        x_head = self.head(x) 
        
        res = x_head

        align_list = []
        res = self.alignment[0](res)+res
        align_list.append(res)

        for id,layer in enumerate(self.body):
            res = layer(res)
            if id in [0,3,7]:
                res = self.alignment[id//2+1](res) + res
                align_list.append(res)

        res = self.fuse_align(torch.cat(align_list,1))
        
        res += x_head       
        
        out = self.tail(res) # [bz,s,h,w]
        
        out[:,::self.args.upscale] = x
        out = out.permute(0,2,3,1).contiguous()
        
        return out


if __name__ == '__main__':
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

    gpy_id = 0
    model = newmodel(args).cuda(gpy_id)
    x = torch.ones(1,args.n_size,args.n_size,args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(1,args.n_size,args.n_size,args.hr_slice_patch).cuda(gpy_id)
    from torchsummary import summary
    summary(model,(256, 256, 4))
    pred=model(x)
    print(pred.shape)