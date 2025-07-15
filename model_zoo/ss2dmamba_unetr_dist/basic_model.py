import torch
import torch.nn as nn
from torch.nn import init
import einops
from functools import partial
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
from einops import rearrange, repeat
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
#from unet import UNet
from .unet import UNet


def make_model(args):
    return newmodel(args)

#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        #d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class I2Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,head_num=1,win_num_sqrt=16,window_size=16):
        super(I2Block, self).__init__()

        self.res_scale = res_scale
        self.unet=UNet(64,64)
        self.unshuffle=nn.PixelUnshuffle(4)
        self.shuffle=nn.PixelShuffle(4)
        '''
        self.mamba= Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=256, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    )
    '''
        self.mamba= SS2D(d_model=1024, dropout=0, d_state=16)
            
    def forward(self, x):
        x_u=self.unet(x)
       # x_inter = self.inter_slice_branch(x).mul(self.res_scale)
        mamba_x=self.unshuffle(x)
        #input=einops.rearrange(mamba_x,'b d h w -> b d (h w)')
        input=einops.rearrange(mamba_x,'b d h w -> b h w d')
        input=input.contiguous()
        output = self.mamba(input)
        #x_mamba=einops.rearrange( output,'b d (h w) -> b d h w',w=mamba_x.shape[-1])
        x_mamba=einops.rearrange( output,'b h w d -> b d h w')
        x_mamba=self.shuffle(x_mamba)
        #out = x_inter + x_mamba + x
        out = x_u + x_mamba + x
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
        '''
        blocks=1
        cblayers=2
        depth=1
        '''
        blocks=3
        cblayers=2
        depth=1
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
        conv=default_conv
        self.head = nn.Sequential(conv(in_slice,n_feats,kernel_size),
                                  nn.ReLU(),
                                  conv(n_feats,n_feats,kernel_size))
        
        modules_body = [
            I2Group(
                conv, n_depth=depth,n_feat=n_feats, kernel_size=kernel_size, act=act, res_scale=res_scale, 
                head_num=head_num, win_num_sqrt=win_num_sqrt,window_size=window_size) for _ in range(blocks)]
        self.body = nn.ModuleList(modules_body)
        
        self.alignment = nn.ModuleList([CrossViewBlock(n_feats) for _ in range(cblayers)])

        self.fuse_align = nn.Conv2d(cblayers*n_feats,n_feats,1,1,0)

        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats,out_slice,kernel_size)]
        self.tail = nn.Sequential(*modules_tail)
        
        self.aux_s = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(32, out_slice, kernel_size=1)
            )
        
    def forward(self, x):
        #print('model_name=mamba_unet_multi')
        x = x.permute(0,3,1,2)
        x = x.contiguous()
        x_head = self.head(x) 
        res = x_head

        align_list = []
        res = self.alignment[0](res)+res
        align_list.append(res)
        id_list=[]
        id_list=[2]
        
        #id_list=[1,3]
        for id,layer in enumerate(self.body):
            res = layer(res)
            if  id==0:
                res_middle=self.aux_s(res)
            if id in id_list:
                res = self.alignment[id//2](res) + res
                #res = self.alignment[id+1](res) + res
                align_list.append(res)

        res = self.fuse_align(torch.cat(align_list,1))
        
        res += x_head       
        
        out = self.tail(res) # [bz,s,h,w]
        
        out[:,::self.args.upscale] = x
        out = out.permute(0,2,3,1).contiguous()
        
        return out,res_middle


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
    #x = torch.ones(1,args.n_size,args.n_size,args.lr_slice_patch).cuda(gpy_id)
    #y = torch.ones(1,args.n_size,args.n_size,args.hr_slice_patch).cuda(gpy_id)
    x=torch.randn(1,256,256,4).cuda(gpy_id)
    from torchsummary import summary
    from thop import profile
    #summary(model,(256, 256, 4))
    pred=model(x)
    print(pred.shape)
    #flops,params=profile(model,(x,))
    #print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))