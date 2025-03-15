"""
多头神经网络模型实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple

class AttentionModule(nn.Module):
    """注意力模块，用于增强特征提取"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        
        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention = self.spatial_attention(spatial_input)
        
        # 应用空间注意力
        x = x * spatial_attention
        
        return x

class PrintQualityHead(nn.Module):
    """打印质量分类头"""
    
    def __init__(self, in_features: int, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ParameterRegressionHead(nn.Module):
    """打印参数回归头"""
    
    def __init__(self, in_features: int, num_parameters: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_parameters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DefectDetectionHead(nn.Module):
    """缺陷检测头"""
    
    def __init__(self, in_features: int, num_defect_types: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_defect_types)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiHeadPrintQualityModel(nn.Module):
    """多头神经网络模型用于3D打印质量评估"""
    
    def __init__(
        self, 
        backbone_name: str = 'resnet50', 
        pretrained: bool = True,
        num_quality_classes: int = 4,
        num_parameters: int = 3,
        num_defect_types: int = 5,
        use_attention: bool = True
    ):
        super().__init__()
        
        # 加载主干网络
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True
        )
        
        # 获取主干网络的特征维度
        dummy_input = torch.zeros(1, 3, 224, 224)
        features = self.backbone(dummy_input)
        last_feature_map = features[-1]
        feature_dim = last_feature_map.shape[1]
        
        # 注意力模块（可选）
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(feature_dim)
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征维度
        flattened_dim = feature_dim
        
        # 各输出头
        self.quality_head = PrintQualityHead(flattened_dim, num_quality_classes)
        self.parameter_head = ParameterRegressionHead(flattened_dim, num_parameters)
        self.defect_head = DefectDetectionHead(flattened_dim, num_defect_types)
        
        # 保存注意力图用于可视化
        self.attention_maps = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 获取主干网络的特征
        features = self.backbone(x)
        feature_map = features[-1]
        
        # 应用注意力（如果启用）
        if self.use_attention:
            feature_map = self.attention(feature_map)
            self.attention_maps = feature_map  # 保存用于可视化
        
        # 全局池化
        pooled = self.global_pool(feature_map)
        flattened = torch.flatten(pooled, 1)
        
        # 计算各头的输出
        quality_output = self.quality_head(flattened)
        parameter_output = self.parameter_head(flattened)
        defect_output = self.defect_head(flattened)
        
        # 返回所有输出
        return {
            "quality": quality_output,
            "parameters": parameter_output,
            "defects": defect_output
        }
    
    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """获取注意力图用于可视化"""
        return self.attention_maps

# 为Apple M系列芯片优化的配置函数
def optimize_for_apple_silicon(model: nn.Module) -> nn.Module:
    """为Apple Silicon芯片优化模型"""
    # 确保使用MPS后端（如果可用）
    if torch.backends.mps.is_available():
        model = model.to(torch.device("mps"))
        print("使用MPS后端进行加速")
    else:
        print("MPS后端不可用，使用CPU")
    
    return model

# 为NVIDIA CUDA优化的配置函数
def optimize_for_cuda(model: nn.Module) -> nn.Module:
    """为NVIDIA CUDA优化模型"""
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        print(f"使用CUDA后端进行加速: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA后端不可用，使用CPU")
    
    return model

# 创建模型的工厂函数
def create_model(config: Dict) -> nn.Module:
    """根据配置创建模型"""
    model = MultiHeadPrintQualityModel(
        backbone_name=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        num_quality_classes=config['model']['num_quality_classes'],
        num_parameters=config['model']['num_parameters'],
        num_defect_types=config['model']['num_defect_types'],
        use_attention=config['model']['use_attention']
    )
    
    # 根据平台优化模型
    if config['training']['device'] == 'mps':
        model = optimize_for_apple_silicon(model)
    elif config['training']['device'] == 'cuda':
        model = optimize_for_cuda(model)
    else:
        model = model.to(torch.device("cpu"))
        print("使用CPU进行计算")
    
    return model
