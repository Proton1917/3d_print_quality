import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ViTBackbone(nn.Module):
    """Vision Transformer骨干网络，用于特征提取
    使用PyTorch原生实现，不依赖timm库"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, 
                 pretrained=True, freeze_layers=True):
        """
        初始化Vision Transformer骨干网络
        
        参数:
            img_size: 输入图像大小
            patch_size: 图像分块大小
            in_channels: 输入图像通道数
            embed_dim: 嵌入维度
            depth: Transformer编码器层数
            num_heads: 多头注意力的头数
            mlp_ratio: MLP隐藏层维度比例
            pretrained: 是否使用预训练权重（注：此简化版本不支持预训练权重加载）
            freeze_layers: 是否冻结主干网络参数
        """
        super(ViTBackbone, self).__init__()
        
        # 计算特征维度
        self.features_dim = embed_dim
        
        # 创建图像分块嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=depth)
        
        # 初始化参数
        self._init_weights()
        
        # 冻结backbone参数
        if freeze_layers:
            for param in self.parameters():
                param.requires_grad = False
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """前向传播"""
        # 图像分块嵌入
        x = self.patch_embed(x)
        
        # 添加分类token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过Transformer编码器
        x = self.transformer(x)
        
        # 只返回分类token的输出作为特征
        return x[:, 0]


class PatchEmbedding(nn.Module):
    """将图像分割为固定大小的块并进行线性投影"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        """
        x: [B, C, H, W]
        输出: [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"输入图像尺寸应为 {self.img_size}x{self.img_size}"
        
        # 将图像分块并投影为嵌入向量
        x = self.projection(x)  # [B, embed_dim, grid_size, grid_size]
        x = x.flatten(2)  # [B, embed_dim, grid_size*grid_size]
        x = x.transpose(1, 2)  # [B, grid_size*grid_size, embed_dim]
        
        return x


def get_vit_backbone(img_size=224, patch_size=16, in_channels=3, 
                    embed_dim=768, depth=12, num_heads=12, 
                    pretrained=True, freeze_layers=True):
    """获取Vision Transformer骨干网络实例"""
    return ViTBackbone(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        pretrained=pretrained,
        freeze_layers=freeze_layers
    )