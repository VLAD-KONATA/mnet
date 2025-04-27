import torch
import torch.nn as nn
from torch.nn import init
import einops
from functools import partial
import torch.nn.functional as F
from einops.layers.torch import Rearrange


from .dct_util import DCT2x,IDCT2x
#from dct_util import DCT2x,IDCT2x
from .utils_win import window_partitionx,window_reversex
from mamba_ssm import Mamba

def make_model(args):
    return newmodel(args)

#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

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
    
pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class IntraSliceBranch(nn.Module):
    def __init__(self,conv=nn.Conv2d,n_feat=64,kernel_size=3,bias=True,
                 head_num=1,win_num_sqrt=16,window_size=16):
        super().__init__()
        
        self.win_num_sqrt = win_num_sqrt
        self.window_size = window_size

        self.dct = DCT2x()
        self.norm = nn.LayerNorm(n_feat)
        self.conv = nn.Sequential(
            conv(n_feat,n_feat,1, bias=bias),
            conv(n_feat,n_feat,3,1,1, bias=bias),
            conv(n_feat,n_feat,3,1,1, bias=bias)
        )
        self.idct = IDCT2x()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.attn = nn.Sequential(
            PreNormResidual(dim=n_feat, fn=FeedForward(dim=win_num_sqrt**2, expansion_factor=1, dropout=0, dense=chan_first)), # dim=num_patch
            PreNormResidual(dim=n_feat, fn=FeedForward(dim=n_feat, expansion_factor=2, dropout=0, dense=chan_last)) # dim=h*w*c 
        )
        self.last_conv = conv(n_feat,n_feat,kernel_size=1,bias=bias)

    def forward(self,x):
        b,c,h,w = x.shape
        x_dct = self.dct(x)
        x_dct = einops.rearrange(x_dct,'b c h w -> b (h w) c')
        x_dct = self.norm(x_dct)
        x_dct = einops.rearrange(x_dct,'b (h w) c -> b c h w',h=h,w=w)
        x_dct = self.conv(x_dct)

        x_dct_windows = window_partitions(x_dct,window_size=h//self.win_num_sqrt) # [b,c,h,w]
     
        bi,ci,hi,wi = x_dct_windows.shape
        x_dct_windows = einops.rearrange(x_dct_windows,'b c h w -> b (h w) c')
        x_dct_windows_attn = self.attn(x_dct_windows)
        x_dct_windows = x_dct_windows + x_dct_windows_attn
        x_dct_windows = einops.rearrange(x_dct_windows,'b (h w) c -> b c h w',h=hi,w=wi)
        
        x_dct_attn =  window_reverses(x_dct_windows,window_size=h//self.win_num_sqrt,H=h,W=w)

        x_dct_idct = self.idct(x_dct_attn)
        x_attn = self.last_conv(x_dct_idct)

        return x_attn        
        

#m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items=keys
        
class newmodel(nn.Module):
    def __init__(self,args=None,conv=default_conv):
        super(newmodel, self).__init__()
        self.args = args
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
        self.intra_slice= IntraSliceBranch(conv=nn.Conv2d,n_feat=n_feats,kernel_size=kernel_size,bias=True
                                  ,head_num=head_num,win_num_sqrt=win_num_sqrt,window_size=window_size)
        self.mamba= Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=256, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    )
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
        
        res1 = self.crossview1(res1)+res1
        align_list.append(res1)
        
        #x_head=self.wide(x_head,keys,train)
        mamba_res=res1
        mamba_res=einops.rearrange(mamba_res,'b d h w -> (b d) h w')
        
        #x_head=self.wide(x_head,keys,train)
        if train:
            output1=self.mamba(mamba_res)
            output1=einops.rearrange(output1,'(b d) h w -> b d h w',b=x.shape[0])
            #output1 = self.model1(res1, keys, train)
            output2=self.intra_slice(res1)
        else:
            output1=self.mamba(mamba_res)
            output1=einops.rearrange(output1,'(b d) h w -> b d h w',b=x.shape[0])
            #output1 = self.model1(res1, keys, train)
            output2=self.intra_slice(res1)
            
        output=output1+output2+res1
        res2=self.crossview2(output)+output
        align_list.append(res2)
        
        res=self.fuse_align(torch.cat(align_list,1))
        
        res+=x_head
        out= self.tail(res)
        
        out[:,::self.args.upscale] = x
        out = out.permute(0,2,3,1).contiguous()
        if train:
            return out
        else:
            return out
            #return out
            



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