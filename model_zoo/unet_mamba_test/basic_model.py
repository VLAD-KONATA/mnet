import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops.layers.torch import Rearrange
import einops

class DownBlock(nn.Module):
    def __init__(self,n_feat,res_scale=1):
        super(DownBlock,self).__init__()
        UNetBlock=UNet._block()

def make_model(args):
    return UNet(args)

class UNet(nn.Module):
    def __init__(self, args=None,in_channels=4, out_channels=7, init_features=64):
        super(UNet, self).__init__()
        '''
        self.args = args
        in_channels=args.in_channels
        out_channels=args.out_channels
        init_features=args.init_features
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
        '''
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mamba1=Mamba(d_model=256,d_state=16,d_conv=4,expand=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mamba2=Mamba(d_model=128,d_state=16,d_conv=4,expand=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mamba3=Mamba(d_model=64,d_state=16,d_conv=4,expand=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mamba4=Mamba(d_model=32,d_state=16,d_conv=4,expand=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x,a=None):
        x=x.permute(0,3,1,2)
        x=x.contiguous()
        enc1 = self.encoder1(x)
        mamba1=einops.rearrange(enc1,'b d h w -> (b d) h w')
        mamba1=self.mamba1(mamba1)
        mamba1=einops.rearrange(mamba1,'(b d) h w -> b d h w',b=x.shape[0])
        pool1=self.pool1(mamba1)
        
        #einops.rearrange(enc1,'b d h w -> (b d) h w')
        enc2 = self.encoder2(pool1)
        mamba2=einops.rearrange(enc2,'b d h w -> (b d) h w')
        mamba2=self.mamba2(mamba2)
        mamba2=einops.rearrange(mamba2,'(b d) h w -> b d h w',b=x.shape[0])
        pool2=self.pool2(mamba2)
        
        enc3 = self.encoder3(pool2)
        mamba3=einops.rearrange(enc3,'b d h w -> (b d) h w')
        mamba3=self.mamba3(mamba3)
        mamba3=einops.rearrange(mamba3,'(b d) h w -> b d h w',b=x.shape[0])
        pool3=self.pool3(mamba3)
        
        enc4 = self.encoder4(pool3)
        mamba4=einops.rearrange(enc4,'b d h w -> (b d) h w')
        mamba4=self.mamba4(mamba4)
        mamba4=einops.rearrange(mamba4,'(b d) h w -> b d h w',b=x.shape[0])
        pool4=self.pool4(mamba4)
        
        bottleneck = self.bottleneck(pool4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, mamba4), dim=1)
        dec4=einops.rearrange(dec4,'b d h w -> (b d) h w')
        dec4=self.mamba4(dec4)
        dec4=einops.rearrange(dec4,'(b d) h w -> b d h w',b=x.shape[0])
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, mamba3), dim=1)
        dec3=einops.rearrange(dec3,'b d h w -> (b d) h w')
        dec3=self.mamba3(dec3)
        dec3=einops.rearrange(dec3,'(b d) h w -> b d h w',b=x.shape[0])
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, mamba2), dim=1)
        dec2=einops.rearrange(dec2,'b d h w -> (b d) h w')
        dec2=self.mamba2(dec2)
        dec2=einops.rearrange(dec2,'(b d) h w -> b d h w',b=x.shape[0])
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, mamba1), dim=1)
        dec1=einops.rearrange(dec1,'b d h w -> (b d) h w')
        dec1=self.mamba1(dec1)
        dec1=einops.rearrange(dec1,'(b d) h w -> b d h w',b=x.shape[0])
        dec1 = self.decoder1(dec1)
        out= self.conv(dec1)
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

# 测试网络
if __name__ == "__main__":
    # 输入张量 (batch_size, channels, height, width)
    #x = torch.randn((1, 3, 256, 256))
    
    # 初始化模型
    model = UNet(in_channels=4, out_channels=7).to('cuda:0')
    from torchsummary import summary
    summary(model,(256,256,4))
    # 前向传播
    #output = model(x)
    
    # 输出形状
    #print("Input shape:", x.shape)
    #print("Output shape:", output.shape)