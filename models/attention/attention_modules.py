import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化通道注意力模块
        
        参数:
            in_channels: 输入特征图的通道数
            reduction_ratio: 降维比例
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块
        
        参数:
            kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

class CBAM(nn.Module):
    """CBAM注意力模块: 结合通道和空间注意力"""
    
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        """
        初始化CBAM注意力模块
        
        参数:
            in_channels: 输入特征图的通道数
            reduction_ratio: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class SelfAttention(nn.Module):
    """自注意力模块"""
    
    def __init__(self, in_features, hidden_features=None):
        """
        初始化自注意力模块
        
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度
        """
        super(SelfAttention, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        
        self.query = nn.Linear(in_features, self.hidden_features)
        self.key = nn.Linear(in_features, self.hidden_features)
        self.value = nn.Linear(in_features, self.hidden_features)
        self.out = nn.Linear(self.hidden_features, in_features)
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, in_features)
        batch_size, seq_len = x.size(0), x.size(1)
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 计算注意力分数（缩放点积注意力）
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_features ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        out = self.out(context)
        
        return out
