import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class LearnablePositionEncoding(nn.Module):
    """ 可学习的2D位置编码 """
    def __init__(self, dim, height=256, width=256):
        super().__init__()
        self.height = height
        self.width = width
        self.pos_embed = nn.Parameter(torch.randn(1, dim, height, width) * 0.02)
        
    def forward(self, x):
        return x + self.pos_embed[:, :, :x.size(2), :x.size(3)]

class ChannelProjection(nn.Module):
    """ 通道维度压缩/扩展 """
    def __init__(self, in_ch=64, out_ch=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.proj(x)

class MultiScaleTransformerBlock(nn.Module):
    def __init__(self, dim=32, heads=8, window_sizes=[8, 16, 32]):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_sizes = window_sizes
        self.dim_head = dim // heads
        
        # 共享的QKV投影
        self.to_qkv = nn.Conv2d(dim, dim*3, 3, padding=1, groups=dim//4)
        
        # 多尺度相对位置编码
        self.rel_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(2*ws-1, 2*ws-1, heads) * 0.02)
            for ws in window_sizes
        ])
        
        # 动态卷积融合
        self.fusion = nn.Conv2d(len(window_sizes)*dim, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        outputs = []
        
        for i, ws in enumerate(self.window_sizes):
            # 窗口划分
            if ws < min(H, W):
                x_win = F.avg_pool2d(x, kernel_size=H//ws, stride=H//ws)
            else:
                x_win = x
            
            # 生成QKV
            qkv = self.to_qkv(x_win).chunk(3, dim=1)  # 各[B,C,S,S]
            q, k, v = map(lambda t: rearrange(t, 'b (h d) s1 s2 -> b h (s1 s2) d', 
                          h=self.heads), qkv)
            
            # 相对位置偏置
            rel_pos = self._get_rel_pos(i, q.shape[2]**0.5)
            
            # 缩放点积注意力
            attn = (q @ k.transpose(-2, -1)) * (self.dim_head ** -0.5)
            attn = attn + rel_pos
            attn = attn.softmax(dim=-1)
            
            out = attn @ v  # [B,H,S*S,D]
            out = rearrange(out, 'b h (s1 s2) d -> b (h d) s1 s2', s1=int(q.shape[2]**0.5))
            
            # 上采样恢复尺寸
            if ws < min(H, W):
                out = F.interpolate(out, size=(H,W), mode='bilinear')
            outputs.append(out)
        
        # 多尺度融合
        return self.fusion(torch.cat(outputs, dim=1))
    
    def _get_rel_pos(self, idx, size):
        ws = self.window_sizes[idx]
        rel_pos = self.rel_pos_embeds[idx]
        h = w = int(size)
        return rel_pos[ws-h:ws+h-1, ws-w:ws+w-1].permute(2,0,1)  # [H,S,S]

class ConvFFN(nn.Module):
    """ 卷积增强的前馈网络 """
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 3, padding=1)
        )
        
    def forward(self, x):
        return self.net(x)

class FeatureTransformer(nn.Module):
    def __init__(self, in_ch=64, dim=32, heads=8, depth=3):
        super().__init__()
        # 初始投影
        self.init_proj = ChannelProjection(in_ch, dim)
        self.pos_enc = LearnablePositionEncoding(dim)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(dim),
                MultiScaleTransformerBlock(dim, heads),
                nn.BatchNorm2d(dim),
                ConvFFN(dim)
            ) for _ in range(depth)
        ])
        
    def forward(self, x):
        # 通道压缩 + 位置编码
        x = self.init_proj(x)  # [B,64,256,256] -> [B,32,256,256]
        x = self.pos_enc(x)
        
        # 多尺度Transformer处理
        for block in self.blocks:
            residual = x
            x = block[0](x)  # Norm
            x = block[1](x) + residual  # Attention
            residual = x
            x = block[2](x)  # Norm
            x = block[3](x) + residual  # FFN
        return x

# 使用示例
if __name__ == "__main__":
    model = FeatureTransformer()
    x = torch.randn(4, 64, 256, 256)
    out = model(x)
    print(out.shape)  # 输出: [4, 32, 256, 256]