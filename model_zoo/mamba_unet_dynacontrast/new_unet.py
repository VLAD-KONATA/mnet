import torch
import torch.nn as nn
import torch.nn.functional as F

class HighFreqEnhancer(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.hf_extract = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.ReLU(),
            # 高频敏感卷积核
            nn.Conv2d(ch, ch, 3, padding=1, 
                     groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1)  # 逐点卷积融合
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 自适应强度系数

    def forward(self, x):
        hf = x - F.avg_pool2d(x, 3, stride=1, padding=1)  # 高频分量
        enhanced = self.hf_extract(hf)
        return  self.alpha * enhanced  # 残差连接
    

class DetailEnhancedUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3,init_features=64):
        super(DetailEnhancedUNet, self).__init__()
        features=init_features
        # 编码器（加入高频增强）
        self.enc1 = nn.Sequential(
            HighFreqEnhancer(in_ch),
            nn.Conv2d(64,64, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            HighFreqEnhancer(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            HighFreqEnhancer(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            HighFreqEnhancer(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU()
        )
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DetailEnhancedUNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DetailEnhancedUNet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DetailEnhancedUNet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DetailEnhancedUNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DetailEnhancedUNet._block(features * 2, features, name="dec1")

        #self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        
        # 最终输出层（细节强化）
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            HighFreqEnhancer(64),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        bottleneck = self.bottleneck(self.pool(enc4))

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

        return self.final(dec1)

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
    x = torch.randn((1, 64, 256, 256))
    
    # 初始化模型
    model = DetailEnhancedUNet(64,64)
    
    # 前向传播
    output = model(x)
    
    # 输出形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)