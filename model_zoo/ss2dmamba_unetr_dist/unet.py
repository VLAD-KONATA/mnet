import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
from monai.networks.nets import unetr
class ViTEncoder(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=64, dim=768, depth=6, heads=12):
        super().__init__()
        assert image_size % patch_size == 0, '图像尺寸必须能被patch大小整除'
        
        self.patch_size = patch_size
        self.dim = dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
            ]) for _ in range(depth)
        ])
        
    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.to_patch_embedding(x)
        x += self.pos_embedding
        
        for norm1, attn, norm2, ff in self.layers:
            x = norm1(x)
            x = attn(x, x, x)[0] + x
            x = norm2(x)
            x = ff(x) + x
        
        h_p = h // p
        w_p = w // p
        x = rearrange(x, 'b (h w) c -> b c h w', h=h_p, w=w_p)
        
        return x

class UNetR(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, image_size=256, patch_size=16, 
                 dim=768, vit_depth=6, vit_heads=12):
        super(UNetR, self).__init__()
        
        # 调整初始卷积输出通道数，使后续尺寸匹配
        self.initial_conv = nn.Conv2d(in_channels, dim // 4, kernel_size=3, padding=1)
        
        # 编码器
        self.encoder = ViTEncoder(image_size, patch_size, dim // 4, dim, vit_depth, vit_heads)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 调整通道数确保尺寸匹配
        self.upconv4 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.decoder4 = self._build_decoder_block(dim, dim // 2)  # 输入是上采样后的dim//2 + 编码器的dim
        
        self.upconv3 = nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2)
        self.decoder3 = self._build_decoder_block(dim // 2, dim // 4)
        
        self.upconv2 = nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2)
        self.decoder2 = self._build_decoder_block(dim // 4, dim // 8)
        
        self.upconv1 = nn.ConvTranspose2d(dim // 8, dim // 16, kernel_size=2, stride=2)
        self.decoder1 = self._build_decoder_block(dim // 8, dim // 16)
        
        # 最终输出
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim // 16, dim // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 16, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # 初始投影
        x_proj = self.initial_conv(x)
        
        # 编码器
        enc_features = self.encoder(x_proj)
        
        # 瓶颈层
        bottleneck = self.bottleneck(enc_features)
        
        # 解码器
        # 第一层上采样
        dec4 = self.upconv4(bottleneck)
        # 调整编码器特征尺寸以匹配
        enc_features_resized = F.interpolate(enc_features, scale_factor=2, mode='bilinear', align_corners=True)
        dec4 = torch.cat((dec4, enc_features_resized), dim=1)
        dec4 = self.decoder4(dec4)
        
        # 后续层
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1) + x
    
    @staticmethod
    def _build_decoder_block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

# 测试网络
if __name__ == "__main__":
    # 创建随机输入张量 (B, 64, 256, 256)
    batch_size = 1
    input_tensor = torch.randn(batch_size, 64, 256, 256)
    model=unetr.UNETR(in_channels=64, out_channels=64,img_size=256)
    # 初始化UNetR模型
    #model = UNetR(in_channels=64, out_channels=64)
    
    # 前向传播
    output = model(input_tensor)
    
    # 检查输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证形状匹配
    assert output.shape == input_tensor.shape, "输出形状与输入形状不匹配"
    print("网络测试通过!")