import torch
import torch.nn as nn

class I2Block(nn.Module):
    def __init__(
        self, inch,outch, kernel_size=3,id=0,res_scale=1):
        super(I2Block, self).__init__()
        self.wave=MBWTConv2d(inch,outch,kernel_size, wt_levels=1, ssm_ratio=1, forward_type="v052d",)
        self.dwconv=DWConv2d_BN_ReLU(inch,outch,kernel_size)
        self.res_scale = res_scale

        #self.unet=UNet(64,64)
        self.mamba= Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=(256/(2**(id-1)))**2, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    )

    def forward(self, x):
       # x = self.ffn0(self.dw0(x))
        x_dw=self.dwconv(x)
        x_wave=self.wave(x)
        #x_u=self.unet(x)
        mamba_x=einops.rearrange(x,'b d h w -> b d (h w)')
        mamba_x=mamba_x.contiguous()
        output = self.mamba(mamba_x)
        x_mamba=einops.rearrange(output,'b d (h w) -> b d h w',w=x.shape[-1])
        
        #out = x_dw + x_wave + x_mamba + x
        out = x_dw +x_wave+ x_mamba + x
        #out = x_dw + x_wave  + x
        #out=x_u+x
        #out= self.ffn1(self.dw1(out))
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(UNet, self).__init__()
        
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.encoder3 = nn.Sequential(UNet._block(features * 2, features * 4, name="enc3"),
                                      I2Block(features * 4, features * 4,3,3))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.encoder4 = nn.Sequential(UNet._block(features * 4, features * 8, name="enc4"),
                                      I2Block(features * 8, features * 8,3,4))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        self.bottleneck = nn.Sequential(UNet._block(features * 8, features * 16, name="bottleneck"),
                                        I2Block(features * 16, features * 16,3,5))

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        #self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.decoder4 = nn.Sequential(UNet._block(features * 16, features * 8, name="dec4"),
                                      I2Block(features * 8, features * 8,3,4))
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        #self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.decoder3 = nn.Sequential(UNet._block(features * 8, features * 4, name="dec3"),
                                      I2Block(features * 8, features * 4,3,3))
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

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

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
        super(MBWTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        #self.global_atten =SS2D(d_model=in_channels, d_state=1,
        #     ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        #x = self.base_scale(self.global_atten(x))
        #x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model
    
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
        

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
        

# 测试网络
if __name__ == "__main__":
    # 输入张量 (batch_size, channels, height, width)
    x = torch.randn((1, 3, 256, 256))
    
    # 初始化模型
    model = UNet(in_channels=3, out_channels=3)
    
    # 前向传播
    output = model(x)
    
    # 输出形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)