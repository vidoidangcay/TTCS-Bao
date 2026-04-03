# cbam.py
import torch
import torch.nn as nn

# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels//reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b,c))
        max_out = self.mlp(self.max_pool(x).view(b,c))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b,c,1,1)
        return x * out

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        out = self.sigmoid(out)
        return x * out

# CBAM Block
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# Test CBAM
if __name__ == "__main__":
    cbam = CBAMBlock(64)
    x = torch.randn(4,64,12,12)
    out = cbam(x)
    print(out.shape)  # [4,64,12,12]