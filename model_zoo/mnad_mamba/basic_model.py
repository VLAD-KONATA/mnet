import torch
import torch.nn as nn
from torch.nn import init
import einops
from functools import partial
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .Memory import Memory
from .Reconstruction import convAE
from mamba_ssm import Mamba

#from .dct_util import DCT2x,IDCT2x
#from dct_util import DCT2x,IDCT2x
# from .utils_win import window_partitionx,window_reversex

def make_model(args):
    return newmodel(args)

#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)
    
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

        
        
#m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items=keys
        
class newmodel(nn.Module):
    def __init__(self,args=None,conv=default_conv):
        super(newmodel, self).__init__()
        self.args=args
        upscale = 2
        n_feats = 64
        kernel_size = 3
        res_scale = 1
        num_blocks = 16
        lr_slice_patch = 4
        hr_slice_patch = (lr_slice_patch-1)*upscale + 1
        head_num = 1
        win_num_sqrt = 16
        window_size = 16
        n_size = 256
        in_slice=lr_slice_patch
        out_slice=hr_slice_patch
        self.head = nn.Sequential(conv(in_slice,n_feats,kernel_size),
                                  nn.ReLU(),
                                  conv(n_feats,n_feats,kernel_size))
        self.crossview1 = CrossViewBlock(n_feats)
        self.crossview2 = CrossViewBlock(n_feats)
        self.model1=convAE(n_channel =64,  t_length = 2, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1)
        self.mamba= Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=256, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    )
        
        self.narrow=None
        self.sr=None
        self.fuse_align = nn.Conv2d(2*n_feats,n_feats,1,1,0)
        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats,out_slice,kernel_size)]
        self.tail = nn.Sequential(*modules_tail)
    
    def forward(self,x,keys,train=True):
        align_list = []
        keys= F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1).cuda() # Initialize the memory items=keys
        
        x=einops.rearrange(x,'b h w d-> b d h w')
        x = x.contiguous()
        x_head = self.head(x)
        res1=x_head
        
        res1 = self.crossview1(res1)
        align_list.append(res1)
        #x_head=self.wide(x_head,keys,train)
        mamba_res=res1
        mamba_res=einops.rearrange(mamba_res,'b d h w -> (b d) h w')
        output2=self.mamba(mamba_res)
        output2=einops.rearrange(output2,'(b d) h w -> b d h w',b=x.shape[0])
        if train:
            #output1, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.model1(res1, keys, train)
            output1 = self.model1(res1)

        else:
            #output1, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.model1(res1, keys, train)
            output1 = self.model1(res1)
            
        output=output1+output2+res1

        res2=self.crossview2(output)+res1
        align_list.append(res2)
        

        res=self.fuse_align(torch.cat(align_list,1))
        
        res=res+x_head+res1
        out= self.tail(res)
        
        out[:,::self.args.upscale] = x
        out = out.permute(0,2,3,1).contiguous()
        return out

        if train:
            return out,fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
        else:
            return out,fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss




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
    m_items = F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1).cuda() # Initialize the memory items=keys

    gpy_id = 0
    model = newmodel(args).cuda(gpy_id)
    from torchsummary import summary
    summary(model,(256, 256, 4))